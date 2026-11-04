# Stage one: proposal-aligned three-state map

Completed 4 November 2026. This stage implements the proposal's occupied/free/unknown map from
posed RGB-D observations. It does not extract a mesh, generate visual features, compress cuboids,
or change the proposed VLM architecture. The proposal itself has not been edited.

## Result and scope

- The full automated suite passes: **92 tests**, zero failures. The one warning is a third-party
  Dash pytest-plugin deprecation, unrelated to map geometry.
- Independent analytic plane/voxel intersection tests agree exactly with the classifier, including
  slanted planes and surfaces exactly on cell boundaries.
- Independently rendered box/wall scenes retain occluded space as unknown; a second camera changes
  the newly observed region to free. Invalid depth and unobserved TSDF samples create no evidence.
- An independent ray/box-intersection audit agrees with saved free/hit counts for **256 sampled cells
  in each real run**, over all 32 and all 276 input frames. This computes 78,848 cell/frame cases;
  the 512 resulting frame-count comparisons have zero discrepancies for either evidence type.
- Fresh fusion of the 32 selected input views reproduces the reused-cache ROI **exactly**: zero
  differences in state labels, TSDF, weights, free counts, or hit counts.

These checks establish implementation correctness under the declared voxel and observation
conventions. They do not establish perfect physical occupancy from noisy sensor data. Input-depth
agreement below is explicitly a consistency diagnostic, not independent ground-truth accuracy.

## What changed

The previous schema-2 classifier used any raw depth endpoint to mark a cell occupied. That was not
the proposal's fused-surface definition. `DenseTSDF.classify()` now uses:

1. **Occupied:** a cell intersecting the supported zero set of the fused TSDF, using observed samples
   with `W > 0`.
2. **Free:** a cell traversed by a valid, guarded depth-ray segment and not marked occupied.
3. **Unknown:** neither surface evidence nor free-ray evidence.

Priority is occupied, then free, then unknown. Negative TSDF by itself does not label an object's
interior occupied. Unknown therefore includes hidden interiors, occluded regions, and gaps in
observation support. It does not mean physically solid or physically empty.

`n_hit` remains available as raw evidence for diagnostics; it no longer decides occupancy. A noisy
hit rejected by fusion does not automatically become free: it becomes free only if there is ray
evidence, and otherwise remains unknown. The minimum mesh weights used in earlier experiments
have no effect on the proposal state map.

The new archive schema is **3**. Old categorical arrays are not silently reinterpreted. The new CLI
can reuse schema-2 TSDF/weights/evidence, but always recomputes the state labels.

## Surface interpolation and boundary convention

TSDF samples are at voxel centers. The surface rasterizer in
`src/volumetric_reasoning/geometry/states.py` uses linear interpolation on observed neighboring
edges, bilinear interpolation on fully observed faces, and trilinear interpolation on fully
observed cubes. Each contributing primitive requires all its own samples to be observed. Thus
two observed neighbors can provide edge-crossing evidence without unnecessarily requiring an
entire eight-sample cube. A diagonal pair separated by unknown samples cannot invent a surface.

An interpolation primitive is split where it crosses voxel boundaries. A multilinear function's
minimum and maximum on each resulting box occur at its corners. Opposite signs establish a zero
inside; exact zeros on included boundaries are also handled. This determines which cells intersect
the interpolated zero set without marching cubes or triangle extraction.

Cells own their lower boundary and exclude their upper boundary: `[i, i+1)` in grid coordinates.
A surface exactly on a shared face belongs to the cell on the higher-index side. Exact-zero samples
need an observed nonzero neighbor; a wholly zero primitive alone does not establish a sign change.
Floating-point input TSDF values are used as stored; the method does not introduce a configurable
"near zero" occupied band.

The ROI includes a one-sample TSDF/weight halo for classification. This preserves a crossing whose
support straddles the requested ROI boundary. The halo is saved as `tsdf_halo` and `weight_halo`;
the exported `state`, `tsdf`, `weight`, `n_free`, and `n_hit` arrays cover only the ROI. Where the source
has no external sample, the halo has `W=0`; it does not extrapolate a surface.

Free evidence uses the existing independently tested ray traversal. Rays stop 2 cm before the measured
endpoint in these runs; the endpoint's own cell is not carved by that ray. This guard is a declared
implementation setting, not a calibrated sensor-uncertainty model. Free means traversed at the
voxel resolution; a single ray is not proof that every continuous point in the cell is empty.

## Analytic validation

`tests/test_states.py` includes 36 plane cases, each also tested with the TSDF sign reversed.
The reference computes analytic plane/AABB intersections from the actual plane equation, independently
of the rasterizer's interpolation stencils. Cases cover axial and slanted planes, grid-face and
sample-center alignment, observed-only support, singleton grid dimensions, and an exact trilinear
field with intersecting zero planes. Other tests check finite input validation and a pair of noisy
depth observations fusing into a surface between the raw endpoints.

`tests/test_state_scenes.py` creates depth images by analytic intersections with a box and a wall,
without calling production backprojection, traversal, TSDF, or mesh routines. The end-to-end map
checks free foreground, an occupied observed face, unknown box interior, an occluded unknown region,
a second view revealing free space, and unknown space behind the wall.

![Analytic box and wall visibility check](../outputs/stage_one_validation/synthetic_occlusion.png)

The circled region is unknown from the first camera and free after the second camera observes it.
The dashed outlines are the known synthetic geometry, not a reconstructed mesh. The synthetic
fixture uses 4 cm cells; the real-data experiment below uses 2 cm cells.

`tests/test_state_map.py` checks the ROI halo, fresh and cached mesh-free exports, state-schema
migration, empty diagnostics, and an independent audit that detects deliberately corrupted counters.
Existing independent segment/AABB tests cover DDA boundary ties, axis-parallel rays, external cameras,
guard distances, and zero-length segments.

## Real ScanNet++ region

Scene: `1ada7a0617`. Requested ROI: `(1, 1, 0.1)` to `(3, 3, 2.5)` metres. Reuse of the existing
2 cm lattice snaps those bounds outward, producing:

- Origin: `(0.9823271408750444, 0.9803117086749398, 0.09145765909062947)` m.
- Upper bound: `(3.0023271408750443, 3.00031170867494, 2.5114576590906292)` m.
- Size: **2.02 × 2.02 × 2.42 m**, **101 × 101 × 121 = 1,234,321 cells**.
- TSDF truncation: 8 cm. Free-ray guard: 2 cm. Valid depth: `(0.1, 4.0]` m.

The 32-view input is the existing uniform selection; the 276-view input is the existing stride-20
selection. They are separate observation sets, not a nested sequence. Their exact frame IDs are in
the manifests. Both use the same region and voxel lattice; each map only uses its own selected
depth observations. The source room bounds came from the previous offline comparison, as recorded
in the nested source manifest. No laser reference contributes to these state maps or this audit.

| Input views | Occupied | Free | Unknown | Unknown share |
|---:|---:|---:|---:|---:|
| 32 | 16,873 | 1,102,101 | 115,347 | 9.345% |
| 276 | 23,016 | 1,182,867 | 28,438 | 2.304% |

These are **voxel counts, not decoder-token counts**. No cuboid compression has been performed.
The fine grid has exactly three states; the old coarse-pooling diagnostic's MIXED code is absent.

![Real-data state comparison](../outputs/stage_one_validation/comparison_states.png)

The larger input set observes more space beneath/around the furniture. The gray areas in the
32-view map are a legitimate representation of missing evidence, not an error to fill in.

### Input-depth consistency

All valid depth pixels in every selected view contribute to the point reference. Surface cells are
represented by their centers, which introduces up to 1.73 cm of voxel quantization distance at 2 cm
resolution. Point-distance diagnostics use at most 100,000 deterministic probes in each direction.

| Diagnostic | 32 views | 276 views |
|---|---:|---:|
| Median surface-cell-center to nearest input-depth point | 0.77 cm | 0.42 cm |
| 90th percentile of that distance | 1.88 cm | 1.22 cm |
| Surface-cell centers within 5 cm of an input point | 99.48% | 99.93% |
| Median input endpoint to nearest occupied-cell center | 1.12 cm | 1.34 cm |
| Input endpoints within 5 cm of an occupied-cell center | 96.49% | 94.13% |
| Free cells with no traversal evidence | 0 | 0 |
| Occupied cells with unobserved center samples | 0 | 0 |
| Unknown cells with free-ray evidence | 0 | 0 |

The larger observation set also contains additional inconsistent/noisy endpoints; more views do not
require every input-consistency statistic to improve. These distances are not occupancy precision
or recall and do not measure completeness of the physical room.

A stricter projection diagnostic flags 10 of 16,873 surface-cell centers in the 32-view run as lying
more than 5 cm in front of every valid depth measurement at their projected center pixels; the
276-view run has zero such flags. All ten centers are within 1.16–3.33 cm of measured points and
have only one or two valid projected views. A cell intersecting a surface need not have its center
on that surface, and pixel-depth discontinuities can make the center projection disagree strongly.
These are retained as diagnostic flags, not silently filtered with a new confidence rule. Details
are in `uniform_32/depth_support_flags.json`.

### Independent ray audit and repeatability

The geometric audit selects cells across unknown/free/occupied states and raw hit/traversal overlap.
For every selected cell and every input view, it uses original depth pixels, inverse intrinsics,
camera poses, and explicit ray/cell slab intersections. It does not call production DDA or production
backprojection. Computed frame-level support counts match the saved counters in all sampled cells.
This is an independent computational check of the same observations, not additional sensor truth.

Fresh 32-view fusion was then run with the exact exported ROI bounds and a one-cell support halo.
Its state, TSDF, weights, and both evidence counters are bit-for-bit equal to the cached-volume
result. This checks that cropping and reuse have not introduced classification or traversal errors.

## Outputs and commands

- `outputs/stage_one_validation/uniform_32/`: primary limited-view map, slices, summary and manifest.
- `outputs/stage_one_validation/stride_20/`: same-ROI dense-observation comparison.
- `outputs/stage_one_validation/fresh_32/`: independently re-fused 32-view repeatability run.
- `outputs/stage_one_validation/validation.json`: independent audit and fresh/cache comparisons.
- `outputs/stage_one_validation/tests.xml`: full automated test results.
- Each real map's `ray_audit.json` records the exact sampled cell indices and any mismatches.

Recompute labels from the previous saved reconstruction (writes a new timestamped directory):

```bash
PYTHONPATH=src venv/bin/python -m volumetric_reasoning.state_map \
  --source-volume outputs/reconstruction_1ada7a0617_20261103T221527262540Z/uniform_32/volume.npz \
  --bounds 1 1 0.1 3 3 2.5
```

For a fresh 32-view map, independently anchored at the requested minimum:

```bash
PYTHONPATH=src venv/bin/python -m volumetric_reasoning.state_map \
  --data-root data --scene 1ada7a0617 --frames 32 \
  --bounds 1 1 0.1 3 3 2.5 \
  --voxel-size 0.02 --truncation 0.08 --free-margin 0.02
```

To reproduce the exact grid of the reported fresh/cache comparison, substitute the full-precision
origin and upper bound listed above. A fresh grid at `(1,1,0.1)` has a slightly different lattice,
so its cell counts need not match the cached grid. `--output` requires a new or empty directory.

Rerun verification:

```bash
PYTHONPATH=src venv/bin/python -m pytest -q
PYTHONPATH=src venv/bin/python documents/validate_stage_one.py
```

The second command audits the already-produced named runs and regenerates the analytic occlusion
figure. It accepts `--run-dir` for a different set of matching run directories. It does not modify
their voxel maps. Manifests record source-code hashes and the reused input archive hash.

## Remaining scope

This stage is complete under the proposal's observation-based definition. Sensor/pose error,
finite ray sampling, truncation, and TSDF interpolation at depth discontinuities remain physical
and representational limitations. A good-looking slice or passing test cannot prove every real
voxel physically correct. No learned completion or confidence-based replacement of the proposal's
`W > 0` rule was introduced. The next proposal stage is surface-feature max pooling and separate
homogeneous free/unknown cuboid construction; neither has been started here.
