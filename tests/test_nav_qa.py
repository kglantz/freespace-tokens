"""Checks of nav_qa.py on hand-built rooms with known answers (under a minute, no data needed).

Room: 6 m x 4 m, walls 2 m high. Object A (sofa) at x 0.5-1.0, object B (bed) at x 5.0-5.5, both y 1.5-2.5,
0.8 m high. "At" an object = robot edge within 0.3 m of it, so for the 50 cm travel robot (radius 0.25) the
zones end at x = 1.0 + 0.55 = 1.55 and start at x = 5.0 - 0.55 = 4.45.

Run from the repo root:  python tests/test_nav_qa.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volumetric_reasoning.nav_qa import FloorGraph, Scan, collision_answer, make_questions, shift  # noqa: E402

STEP = 0.02


def box(lo, hi):
    """Points filling an axis-aligned box (2 cm spacing)."""
    axes = [np.arange(l, h + 1e-9, STEP) for l, h in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def room(extra=(), floor=True):
    """Scan of the room plus extra (label, lo, hi) boxes. Instance ids: floor 1, walls 2, A 10, B 11, extras 20+.
    An extra with label "part of bed" is also a member of the bed (overlapping instances, as in ScanNet++)."""
    parts = [("floor", [0, 0, 0], [6, 4, 0] if floor else [0.2, 0.2, 0]),
             ("wall", [0, 0, 0], [0, 4, 2]), ("wall", [6, 0, 0], [6, 4, 2]),
             ("wall", [0, 0, 0], [6, 0, 2]), ("wall", [0, 4, 0], [6, 4, 2]),
             ("sofa", [0.5, 1.5, 0], [1.0, 2.5, 0.8]), ("bed", [5.0, 1.5, 0], [5.5, 2.5, 0.8])] + list(extra)
    ids = [1, 2, 2, 2, 2, 10, 11] + list(range(20, 20 + len(extra)))
    pts, members, objects, n = [], {}, {}, 0
    for (label, lo, hi), i in zip(parts, ids):
        p = box(lo, hi)
        pts.append(p)
        members[i] = np.concatenate([members.get(i, np.zeros(0, int)), np.arange(n, n + len(p))])
        if label == "part of bed":
            members[11] = np.concatenate([members[11], np.arange(n, n + len(p))])
        n += len(p)
        lo, hi = np.array(lo, float), np.array(hi, float)
        objects[i] = {"id": i, "label": label, "obb": {"centroid": ((lo + hi) / 2).tolist(),
                                                       "axesLengths": (hi - lo).tolist()}}
    return Scan(np.concatenate(pts), members, objects)


def travel(scan, a=10, b=11, radius=0.25):
    g = FloorGraph(scan, radius)
    return float(g.distances_from(a)[g.zone(b)].min())


def collision(scan):
    return [collision_answer(scan, 10, 11, w) for w in (0.4, 0.6, 0.8)]


# 0. grid shift
a = np.arange(12).reshape(3, 4)
assert (shift(a, 1, 0, fill=-1) == [[4, 5, 6, 7], [8, 9, 10, 11], [-1, -1, -1, -1]]).all()
assert (shift(a, 0, -1, fill=-1) == [[-1, 0, 1, 2], [-1, 4, 5, 6], [-1, 8, 9, 10]]).all()
print("0. grid shift OK")

# 1. open room: straight line for every width; travel = 4.45 - 1.55 = 2.9 m
s = room()
assert collision(s) == ["yes"] * 3, collision(s)
t_open = travel(s)
assert abs(t_open - 2.9) < 0.1, t_open
print(f"1. open room: collision yes x3, travel {t_open:.2f} m (expected 2.9)")

# 2. not obstacles: a cable on the floor (below 0.15 m) across the room, a ceiling lamp at 1.6-2.0 m;
#    a 10 cm lamp on the centre line is an obstacle, but a straight line beside it still exists
for extra in ([("cable", [2.0, 0.0, 0], [2.1, 4.0, 0.1])], [("ceiling lamp", [2.8, 0.0, 1.6], [3.2, 4.0, 2.0])],
              [("lamp", [2.95, 1.95, 0], [3.05, 2.05, 1.4])]):
    s = room(extra)
    assert collision(s) == ["yes"] * 3, (extra, collision(s))
assert abs(travel(room([("cable", [2.0, 0.0, 0], [2.1, 4.0, 0.1])])) - t_open) < 1e-9
print("2. cable / ceiling lamp ignored; a small lamp can be passed in a straight line")

# 3. door in a partition (x 2.9-3.1) centred on y = 2, width g. A straight line through it needs g / 2 > r:
#    g = 1.2: yes for all widths (r = w/2 + 0.1 <= 0.5 < 0.6);
#    g = 0.5: 80 cm -> no (even r = 0.3 > 0.25), 40 / 60 cm too close to call (r = 0.1 / 0.2 < 0.25 < 0.3 / 0.4)
def door(g):
    return room([("wall", [2.9, 0, 0], [3.1, 2 - g / 2, 2]), ("wall", [2.9, 2 + g / 2, 0], [3.1, 4, 2])])
assert collision(door(1.2)) == ["yes"] * 3, collision(door(1.2))
assert collision(door(0.5)) == [None, None, "no"], collision(door(0.5))
print("3. door 1.2 m: yes x3; door 0.5 m: 80 cm no, 40 / 60 cm not asked")

# 4. partition from y = 0 to 3.4 (0.6 m gap at the top). Collision zones end at y = 2.5 + 0.8 = 3.3, so every
#    straight segment between them stays below y = 3.3 and hits the partition -> no for every width.
#    The path goes through the gap. Exact taut path: from A's zone (radius 0.55 around A's corner (1.0, 2.5))
#    along the tangent to the circle of radius 0.25 around the partition end (2.9, 3.4), over its 0.2 m wide top,
#    symmetrically down to B's zone.
s = room([("partition", [2.9, 0, 0], [3.1, 3.4, 2])])
assert collision(s) == ["no"] * 3, collision(s)
d = np.hypot(1.9, 0.9)
exact = 2 * (np.sqrt(d ** 2 - 0.25 ** 2) - 0.55 + 0.25 * (np.arctan2(0.9, 1.9) + np.arcsin(0.25 / d))) + 0.2
t_wall = travel(s)
assert exact <= t_wall < exact * 1.04, (t_wall, exact)                    # grid: up to ~3% longer
print(f"4. partition with gap: collision no x3, travel {t_wall:.2f} m (exact {exact:.2f} m)")

# 5. gap width for the travel robot (radius 0.25): needs a gap > 0.5 m; a closed partition = unreachable
for gap, passable in [(0.4, False), (0.7, True)]:
    assert np.isfinite(travel(room([("partition", [2.9, 0, 0], [3.1, 4.0 - gap, 2])]))) == passable, gap
assert not np.isfinite(travel(room([("partition", [2.9, 0, 0], [3.1, 4.0, 2])])))
print("5. 0.4 m gap blocked, 0.7 m gap passable, closed partition unreachable")

# 6. overlapping instances: a pillow that is also part of the bed, sticking out to x = 4.6 -> B's zone starts at
#    4.6 - 0.55 = 4.05 and travel = 4.05 - 1.55 = 2.5 m; a rug instead of floor is walkable floor
t = travel(room([("part of bed", [4.6, 1.9, 0.3], [5.0, 2.1, 0.9])]))
assert abs(t - 2.5) < 0.1, t
t = travel(room([("rug", [0, 0, 0], [6, 4, 0.01])], floor=False))
assert abs(t - t_open) < 0.1, t
print("6. parts of B belong to B; rugs are floor")

# 7. endpoints: >= 25 cm, starting below 1.5 m, not structure / unnamed; repeated labels allowed (marked A / B)
s = room([("chair", [2, 3, 0], [2.4, 3.4, 0.9]), ("chair", [4, 3, 0], [4.4, 3.4, 0.9]),
          ("mug", [3, 3, 0.5], [3.1, 3.1, 0.6]), ("tv", [3, 3.5, 1.6], [3.8, 3.6, 2.0]),
          ("table", [2, 0.5, 0], [3, 1.0, 0.75]), ("object", [4, 0.5, 0], [4.5, 1.0, 0.5])])
assert s.endpoints() == [10, 11, 20, 21, 24], s.endpoints()     # sofa, bed, both chairs, table
print("7. endpoint selection OK")

# 8. questions: balanced yes/no, answers reproducible, distances consistent
s = room([("table", [2.8, 2.9, 0], [3.2, 3.3, 0.75]), ("plant", [4.2, 0.8, 0], [4.5, 1.1, 1.2]),
          ("shelf", [1.5, 0.3, 0], [2.5, 0.6, 1.8]), ("partition", [2.9, 0, 0], [3.1, 2.4, 2])])
rows = make_questions(s, "synthetic", np.random.default_rng(0))
col = [r for r in rows if r["task"] == "nav_collision"]
dist = [r for r in rows if r["task"] == "nav_distance"]
assert col
for w in (0.4, 0.6, 0.8):     # balanced for every width
    assert sum(r["answer"] == "yes" for r in col if r["robot_width_m"] == w) == \
        sum(r["answer"] == "no" for r in col if r["robot_width_m"] == w), w
for r in col:
    assert collision_answer(s, r["a"]["id"], r["b"]["id"], r["robot_width_m"]) == r["answer"]
for r1 in col:      # monotone in the width: never "no" for a narrower and "yes" for a wider robot
    for r2 in col:
        if (r1["a"]["id"], r1["b"]["id"]) == (r2["a"]["id"], r2["b"]["id"]) and r1["robot_width_m"] < r2["robot_width_m"]:
            assert not (r1["answer"] == "no" and r2["answer"] == "yes")
for r in dist:
    assert float(r["answer"]) == round(r["travel_m"], 1) and r["travel_m"] >= r["straight_m"] - 1e-6 >= 1.0 - 1e-6
for r in rows:
    assert f"the {r['a']['label']} (A) to the {r['b']['label']} (B)" in r["question"], r["question"]
assert len({r["id"] for r in rows}) == len(rows)
print(f"8. {len(col)} collision (balanced) + {len(dist)} distance questions OK")
print("All nav_qa checks passed.")
