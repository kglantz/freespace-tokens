"""Navigation questions with ground truth from the ScanNet++ laser scan (independent of our iPhone reconstruction).

  nav_collision: "Can a robot {w} cm wide drive in a straight line from the {a} (A) to the {b} (B) without
                  hitting anything? Answer yes or no."                                    -> "yes" / "no"
  nav_distance:  "How many meters does a robot 50 cm wide have to travel to get from the {a} (A) to the {b} (B)?"
                                                                                          -> e.g. "3.4"
(A) and (B) are the two endpoint objects, marked in the model input; the label alone may be ambiguous.

Ground truth from the laser mesh vertices (edges <= 6 cm) and the instance annotations, on a 5 cm grid:
  * floor height = median height of the floor vertices; floor cells = cells with any vertex within 5 cm of the
    floor height (floor, rugs, mats), small gaps closed and enclosed holes filled (the scanner cannot see the floor
    under its own tripod);
  * obstacle = any vertex 0.15-1.5 m above the floor (the robot drives over lower things, under higher ones);
  * walkable(r) = floor cells more than r from every obstacle cell (robot of radius r, centre on the cell);
  * collision: is there a straight segment on walkable cells between the zones of A and B (walkable cells with the
    robot centre within 0.8 m of the object's footprint, the same for every width)? "yes" if there is one even
    for a robot 20 cm wider (r = w/2 + 0.1), "no" if there is none even for one 20 cm narrower (r = w/2 - 0.1),
    otherwise the question is not asked (too close to call at the 16 cm resolution of our tokens). Answers are
    monotone in the width by construction; a pair where the sampled segment ends still break that is not asked;
  * distance: shortest path of a robot of radius 0.25 m over walkable cells between the zones of A and B (walkable
    cells where the robot's edge is within 0.3 m of the object), on a 16-neighbour grid (at most ~3% longer than
    the exact path).
Pairs: both distance zones exist and are at least 1 m apart (straight line), for both question types.
Endpoints: annotated objects at least 25 cm in size, starting less than 1.5 m above the floor, and not structure
(walls, floor, ceiling, frames, curtains, cables, ...) or unnamed ("object", "split", ...).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CELL = 0.05                 # m, ground-truth grid
BAND = (0.15, 1.5)          # m above the floor: what the robot can hit
WIDTHS = (0.4, 0.6, 0.8)    # m, robot widths of the collision questions
MARGIN = 0.1                # m, collision: "yes" must hold for radius + MARGIN, "no" for radius - MARGIN
TRAVEL_WIDTH = 0.5          # m, robot width of the distance questions
REACH = 0.3                 # m, distance: zone = walkable cells where the robot's edge is this close to the object
COLLISION_REACH = max(WIDTHS) / 2 + MARGIN + REACH   # 0.8 m, collision: zone = robot centre this close, any width
MIN_GAP = 1.0               # m, smallest straight-line distance between the two zones
LINE_POINTS = 150           # zone cells tried as segment ends (evenly spaced subset)

FLOOR_LABELS = {"floor", "tiled floor", "shower floor", "floor panel"}
STRUCTURE = FLOOR_LABELS | {
    "wall", "tiled wall", "shower wall", "bathroom wall", "door frame", "doorframe", "window frame", "windowframe",
    "windowsill", "window sill", "window lintel", "sliding door rail", "curtain", "window curtain", "shower curtain",
    "curtain rod", "shower curtain rod", "shower rod", "blinds", "roller blinds", "pipe", "heat pipe", "cable",
    "cables", "cable duct", "cable raceway", "electrical duct", "extension cord", "rug", "carpet", "mat", "floormat",
    "door mat", "doormat", "yoga mat", "stair", "structure", "object", "objects", "REMOVE", "remove", "SPLIT", "split"}

COLLISION_Q = ("Can a robot {w} cm wide drive in a straight line from the {a} (A) to the {b} (B) without hitting "
               "anything? Answer yes or no.")
DISTANCE_Q = "How many meters does a robot {w} cm wide have to travel to get from the {a} (A) to the {b} (B)?"


class Scan:
    """Laser mesh vertices with the annotated instances, and the 5 cm floor / obstacle grids."""

    def __init__(self, points: np.ndarray, members: dict[int, np.ndarray], objects: dict[int, dict]):
        """points (N, 3) world coordinates (z up); members: instance id -> its vertex indices (instances may
        overlap, e.g. a blanket is also part of the bed); objects: instance id -> segGroup (label, obb)."""
        self.points, self.members, self.objects = points, members, objects
        floor = [members[i] for i, g in objects.items() if g["label"] in FLOOR_LABELS]
        if not floor:
            raise ValueError("no floor annotation")
        self.floor_z = float(np.median(points[np.concatenate(floor), 2]))
        self.height = points[:, 2] - self.floor_z
        self.lo = points[:, :2].min(0) - 1.0                           # 1 m border: closing / dilation stay inside
        self.shape = tuple(np.ceil((points[:, :2].max(0) + 1.0 - self.lo) / CELL).astype(int))
        self.cell = np.floor((points[:, :2] - self.lo) / CELL).astype(int)
        self.floor = ndimage.binary_fill_holes(
            ndimage.binary_closing(self.mask(np.abs(self.height) < 0.05), iterations=2))
        self.obstacle = self.mask((self.height > BAND[0]) & (self.height < BAND[1]))
        self._distance = ndimage.distance_transform_edt(~self.obstacle) * CELL   # to the nearest obstacle cell

    @classmethod
    def load(cls, scans_dir) -> "Scan":
        d = Path(scans_dir)
        points = np.asarray(o3d.io.read_triangle_mesh(str(d / "mesh_aligned_0.05.ply")).vertices)
        seg = np.array(json.load(open(d / "segments.json"))["segIndices"])
        groups = json.load(open(d / "segments_anno.json"))["segGroups"]
        assert len(seg) == len(points)
        order = np.argsort(seg, kind="stable")                         # vertices grouped by segment
        last = int(seg.max())
        start = np.searchsorted(seg[order], np.arange(last + 2))
        members = {g["id"]: np.concatenate([order[start[s]:start[s + 1]] for s in g["segments"] if s <= last]
                                           + [np.zeros(0, int)]) for g in groups}
        assert len(members) == len(groups), "duplicate instance ids"
        return cls(points, members, {g["id"]: g for g in groups})

    def mask(self, vertices: np.ndarray) -> np.ndarray:
        """(shape) grid: True where at least one of the selected vertices (bool mask or indices) falls."""
        m = np.zeros(self.shape, bool)
        m[self.cell[vertices, 0], self.cell[vertices, 1]] = True
        return m

    def walkable(self, radius: float) -> np.ndarray:
        """Floor cells whose centre is more than `radius` from every obstacle cell centre."""
        return self.floor & (self._distance > radius)

    def zone(self, walk: np.ndarray, i: int, reach: float) -> np.ndarray:
        """Cells of walk whose centre is within `reach` of object i's footprint."""
        return walk & (ndimage.distance_transform_edt(~self.mask(self.members[i])) * CELL <= reach)

    def endpoints(self) -> list[int]:
        """Instance ids usable as question endpoints (see module docstring)."""
        ids = []
        for i, g in sorted(self.objects.items()):
            label, sel = g["label"], self.members[i]
            if label in STRUCTURE or label.startswith("ceiling") or len(sel) < 30:
                continue
            if max(g["obb"]["axesLengths"]) > 0.25 and self.height[sel].min() < BAND[1]:
                ids.append(i)
        return ids


# --- collision ------------------------------------------------------------------------
def straight_line(walk: np.ndarray, za: np.ndarray, zb: np.ndarray):
    """A straight segment (two cells) entirely on walkable cells between a cell of za and a cell of zb, or None.
    Segment ends: up to LINE_POINTS evenly spaced cells of each zone; the segment is checked every half cell."""
    def ends(z):
        c = np.argwhere(z)
        return c[np.linspace(0, len(c) - 1, min(LINE_POINTS, len(c))).round().astype(int)]
    if not za.any() or not zb.any():
        return None
    a, b = ends(za)[:, None, :], ends(zb)[None, :, :]                   # (na, 1, 2), (1, nb, 2)
    ok = np.ones((a.shape[0], b.shape[1]), bool)
    for f in np.linspace(0, 1, int(np.abs(b - a).max() * 2) + 2):
        p = np.rint(a + f * (b - a)).astype(int)
        ok &= walk[p[..., 0], p[..., 1]]
    if not ok.any():
        return None
    i, j = np.argwhere(ok)[0]
    return a[i, 0], b[0, j]


def collision_segment(scan: Scan, a: int, b: int, radius: float):
    """straight_line between the zones of a and b for a robot of this radius. The zones (centre within
    COLLISION_REACH) are the same for every radius, so a line for a wide robot is also one for a narrow robot."""
    walk = scan.walkable(radius)
    return straight_line(walk, scan.zone(walk, a, COLLISION_REACH), scan.zone(walk, b, COLLISION_REACH))


def collision_answer(scan: Scan, a: int, b: int, width: float) -> str | None:
    if collision_segment(scan, a, b, width / 2 + MARGIN) is not None:
        return "yes"
    if collision_segment(scan, a, b, max(width / 2 - MARGIN, 0.0)) is None:
        return "no"
    return None


# --- travel distance ------------------------------------------------------------------
# 16-neighbourhood: (offset, cells the straight move passes through, which must be walkable too)
MOVES = [((0, 1), []), ((1, 0), []), ((1, 1), [(1, 0), (0, 1)]), ((1, -1), [(1, 0), (0, -1)]),
         ((1, 2), [(0, 1), (1, 1)]), ((2, 1), [(1, 0), (1, 1)]), ((1, -2), [(0, -1), (1, -1)]),
         ((2, -1), [(1, 0), (1, -1)])]


def shift(a: np.ndarray, di: int, dj: int, fill=False) -> np.ndarray:
    """out[i, j] = a[i + di, j + dj] (fill outside the grid)."""
    out = np.full_like(a, fill)
    H, W = a.shape
    out[max(0, -di):H - max(0, di), max(0, -dj):W - max(0, dj)] = a[max(0, di):H - max(0, -di), max(0, dj):W - max(0, -dj)]
    return out


class GridPaths:
    """Shortest paths over the walkable cells of a 2-D grid with the given cell size (m)."""

    def __init__(self, walk: np.ndarray, cell: float):
        self.walk = walk
        self.node = np.full(walk.shape, -1)
        self.node[walk] = np.arange(walk.sum())
        src, dst, w = [], [], []
        for (di, dj), through in MOVES:
            ok = walk & shift(walk, di, dj)
            for mi, mj in through:
                ok &= shift(walk, mi, mj)
            src.append(self.node[ok])
            dst.append(shift(self.node, di, dj, fill=-1)[ok])
            w.append(np.full(ok.sum(), np.hypot(di, dj) * cell))
        n = int(walk.sum())
        self.graph = csr_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(n, n))

    def distances(self, sources: np.ndarray) -> np.ndarray:
        """(shape) travel distance from the nearest source cell (walkable) to every cell, inf = unreachable."""
        d = dijkstra(self.graph, directed=False, indices=self.node[sources & self.walk], min_only=True)
        out = np.full(self.walk.shape, np.inf)
        out[self.walk] = d
        return out


class FloorGraph(GridPaths):
    """Shortest paths of a robot of the given radius over the walkable cells of a scan."""

    def __init__(self, scan: Scan, radius: float):
        super().__init__(scan.walkable(radius), CELL)
        self.scan, self.radius = scan, radius

    def zone(self, i: int) -> np.ndarray:
        """Walkable cells where the robot's edge is within REACH of object i."""
        return self.scan.zone(self.walk, i, self.radius + REACH)

    def distances_from(self, i: int) -> np.ndarray | None:
        """(shape) travel distance from the zone of object i to every cell (inf = unreachable); None if no zone."""
        z = self.zone(i)
        return self.distances(z) if z.any() else None


def straight_distance(za: np.ndarray, zb: np.ndarray, cell: float = CELL) -> float:
    """Smallest straight-line distance between two zones (cell centres)."""
    return float(cKDTree(np.argwhere(za)).query(np.argwhere(zb))[0].min() * cell)


# --- questions ------------------------------------------------------------------------
def make_questions(scan: Scan, scene_id: str, rng: np.random.Generator, max_pairs: int = 0) -> list[dict]:
    """Collision questions (yes / no balanced for every width) and distance questions for up to max_pairs (0 = all) object pairs."""
    graph = FloorGraph(scan, TRAVEL_WIDTH / 2)
    ids = scan.endpoints()
    zones = {i: graph.zone(i) for i in ids}
    pairs = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]
             if zones[a].any() and zones[b].any() and straight_distance(zones[a], zones[b]) >= MIN_GAP]
    if max_pairs and len(pairs) > max_pairs:
        pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), max_pairs, replace=False))]

    def obj(i):
        g = scan.objects[i]
        return {"id": i, "label": g["label"], "obb": g["obb"]}

    yes, no = [], []
    for a, b in pairs:
        answers = [collision_answer(scan, a, b, w) for w in WIDTHS]        # WIDTHS ascending
        n = len(WIDTHS)
        if any(answers[i] == "no" and answers[j] == "yes" for i in range(n) for j in range(i + 1, n)):
            continue    # "no" for a narrower robot but "yes" for a wider one (missed segment ends): not asked
        for w, ans in zip(WIDTHS, answers):
            if ans is not None:
                row = {"task": "nav_collision", "a": obj(a), "b": obj(b), "robot_width_m": w, "answer": ans,
                       "question": COLLISION_Q.format(w=round(w * 100), a=obj(a)["label"], b=obj(b)["label"])}
                (yes if ans == "yes" else no).append(row)
    collision = []          # balanced yes / no for every width, so the width alone does not give the answer away
    for w in WIDTHS:
        y, n_ = [r for r in yes if r["robot_width_m"] == w], [r for r in no if r["robot_width_m"] == w]
        k = min(len(y), len(n_))
        collision += [rows[j] for rows in (y, n_) for j in sorted(rng.choice(len(rows), k, replace=False))]

    distance, from_a = [], {}
    for a, b in pairs:
        if a not in from_a:
            from_a[a] = graph.distances_from(a)
        travel = float(from_a[a][zones[b]].min())
        if not np.isfinite(travel):
            continue
        distance.append({"task": "nav_distance", "a": obj(a), "b": obj(b), "robot_width_m": TRAVEL_WIDTH,
                         "answer": f"{travel:.1f}",
                         "question": DISTANCE_Q.format(w=round(TRAVEL_WIDTH * 100), a=obj(a)["label"], b=obj(b)["label"]),
                         "travel_m": round(travel, 3), "straight_m": round(straight_distance(zones[a], zones[b]), 3)})

    rows = collision + distance
    for n, r in enumerate(rows):
        r.update(id=f"nav/{r['task']}/{scene_id}/{n}", scene_id=scene_id)
    return rows
