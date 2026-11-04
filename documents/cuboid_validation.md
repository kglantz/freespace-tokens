# Proposal cuboid construction: implementation and measured counts

Completed 4 November 2026. This implements the proposal's separate merging of free and unknown
cells into homogeneous axis-aligned cuboids. It consumes the already-validated three-state maps;
the proposal and the input maps are unchanged. No mesh, octree, learned completion, coarsening,
surface-feature extraction, or decoder integration is introduced.

## Result

All four real-data exports reproduce their input voxel labels exactly. There are **zero missing
cells, overlapping cells, mixed-state cuboids, occupied cells included in regions, or round-trip
label differences**. Surface-cell indices and metric descriptors also pass serialized-file checks.

All runs retain the **2 cm grid** and full 2.42 m height. The smaller footprint is selected solely
by centered grid indices, identically for both view sets, without inspecting labels or QA answers.

| Region size [m] | Views | Surface candidates | Free cuboids | Unknown cuboids | All scene slots |
|---|---:|---:|---:|---:|---:|
| 2.02 × 2.02 × 2.42 | 32 | 16,873 | 4,079 | 2,568 | **23,520** |
| 2.02 × 2.02 × 2.42 | 276 | 23,016 | 3,789 | 1,298 | **28,103** |
| 1.02 × 1.02 × 2.42 | 32 | 3,626 | 1,048 | 711 | **5,385** |
| 1.02 × 1.02 × 2.42 | 276 | 7,119 | 1,303 | 320 | **8,742** |

![Scene-slot counts after exact compression](../outputs/cuboid_validation/token_counts.png)

The full 32-view region's **1,217,448 free/unknown cells become 6,647 region descriptors**, about
183× fewer. Free cells compress about 270×; unknown cells about 45×. The full 276-view map's
1,211,305 free/unknown cells become 5,087 regions, about 238× fewer. Free and unknown are never
merged with each other. The surface count is unchanged by this compression.

These are planned **scene-token slots**, using one visual token per occupied cell and one region
token per cuboid. They are not text-token counts, not learned embeddings yet, and exclude question,
answer, and other sequence overhead. Surface feature association is pending; whether every occupied
cell receives usable visual features must be checked at that stage. Surface counts are therefore
candidate counts, not a claim that visual tokens have already been constructed.

## Implication for the next step

The proposal's cuboid representation preserves the three-state information while substantially
reducing the volumetric input. It does not make the full region a small sequence: surface candidates
alone account for 16,873 slots in the 32-view run.

The central **32-view crop at 5,385 slots** is a practical fixture for the next implementation step:
frozen image features, association to surface voxels, and channel-wise max pooling. For an
illustrative scene-only budget of 8,192, this leaves 2,807 scene slots; the same crop with 276 views
requires 8,742 and exceeds that example budget. No model/hardware budget has been selected or
benchmarked, and 8,192 is not asserted to be a universal VLM limit.

This crop is an engineering fixture, not evidence that a 1 m footprint contains enough context for
the project's QA task. Before training, region selection must retain the context required by each
question and be identical across matched ablations. It must not depend on the ground-truth answer.
Keeping the full-region results makes that context-versus-input-size tradeoff explicit.

## Algorithm and exact geometry convention

`geometry/cuboids.py` implements a deterministic greedy partition:

1. Work on free and unknown masks separately. Occupied cells remain untouched.
2. Visit the first available cell in a fixed scan order. Extend a run along the first growth axis,
   extend the run into a rectangle along the second, then extend the rectangle along the third.
   Every new slab must consist entirely of still-unassigned cells of this state.
3. Remove the resulting box from the available mask and continue until every state cell is covered.
4. Try all six axis permutations for each state. Keep the fewest-box partition, resolving equal
   counts lexicographically. Sort the final records canonically for stable serialization.

This is an exact partition, **not an optimal minimum-cardinality partition**. No small component,
thin unknown gap, or noisy boundary is dropped to improve compression. Complicated boundaries can
therefore produce many small cuboids. For example, the full 32-view partition contains 445 singleton
free cuboids and 334 singleton unknown cuboids; both are necessary cells under this partition.

Integer bounds are `[start, stop)` in **x/y/z** order. Given the source origin `o` and cell size `h`,
the physical lower and upper faces are `o + start*h` and `o + stop*h`. The descriptor center is
`o + (start + stop)*h/2`; its extent is `(stop-start)*h`. Centers are not rounded to voxel centers.
All coordinates stay in the input world frame, in metres. Surface centers are `o + (index+.5)*h`.
CSV width/height/depth columns correspond, in order, to x/y/z extents; the archive's `region_extent`
is also explicitly ordered x/y/z.

The full grid is 101 × 101 × 121. The crop is source indices `[25:76, 25:76, 0:121]`, giving
51 × 51 × 121 cells. Its origin is approximately `(1.48232714, 1.48031171, 0.09145766)` m and
upper bound `(2.50232714, 2.50031171, 2.51145766)` m. Full precision is recorded in each manifest.
No TSDF is re-fused or reclassified during cropping: the selected original labels are retained.

![Full-region voxel labels and cuboid cross-sections](../outputs/cuboid_validation/uniform_32/cuboid_slices.png)

The lines show intersections of 3-D cuboids with the displayed slice. Thin stripes can be necessary
because the same 3-D cuboid must remain homogeneous above and below the slice; a large blue patch
in a single slice does not imply one valid cuboid through the entire height.

## Validation

The full test suite passes **137 tests**, including 45 added for cuboids and their CLI. The only
warning is the existing third-party Dash pytest-plugin deprecation. Tests include:

- All 256 binary 2×2×2 masks in every growth order, checked by independent enumeration of cell
  coordinates, not by calling the merger or relying only on summed volumes.
- Random three-state grids, reproducibility, uniform grids, empty state classes, and singleton axes.
- Thin occupied barriers, isolated unknown holes, diagonal separation, and checkerboards.
- Known noncubic boxes verifying axis permutations, metric centers/extents, and half-open bounds.
- Deliberately corrupted boxes verifying that gaps, overlaps, mixed labels, occupied inclusion,
  out-of-bounds indices, and invalid state codes are caught.
- CLI state-schema rejection, exact crops, input preservation, serialized descriptors, budget
  accounting, and all four ablations.

Real-data validation expands every box into a coverage grid, counts coverage, and checks the label
of every covered cell against the input. Uncovered non-occupied cells and covered occupied cells
are failures. This catches errors that equal total volume alone would miss. The saved archive is
reloaded and checked again. A separate experiment audit recomputes metric faces directly from the
original source grid, verifies source/output hashes, checks all surface indices, and recounts tokens.

Across two full and two cropped maps, this checks **3,098,084 voxel instances**. All checks pass.
This establishes that compression preserves the input map, not that the underlying noisy depth
observations are physical ground truth; the stage-one report covers that distinction.

## Matched ablation counts

| Region / views | Surface only | Surface + free | Surface + unknown | Surface + both |
|---|---:|---:|---:|---:|
| Full / 32 | 16,873 | 20,952 | 19,441 | 23,520 |
| Full / 276 | 23,016 | 26,805 | 24,314 | 28,103 |
| Crop / 32 | 3,626 | 4,674 | 4,337 | 5,385 |
| Crop / 276 | 7,119 | 8,422 | 7,439 | 8,742 |

Within each row all variants retain the same occupied cells and observation set. Region construction
does not depend on which ablation later consumes it. The 32- and 276-view sets are the stage-one
selections; they are separate sets, not a nested sequence.

## Outputs and reproducibility

Each directory under `outputs/cuboid_validation/` contains:

- `token_geometry.npz`: region integer bounds, state codes, centers/extents, surface indices/centers,
  grid origin/dimensions/scale, frame indices, and schema identifiers. Region schema is 1; source
  state schema is 3. `region_state` uses unknown=0, free=1; occupied=2 is excluded from cuboids.
- `regions.csv`: readable geometry descriptors for every cuboid.
- `cuboid_slices.png`: complete mid-height and vertical cross-sections, with region boundaries.
- `summary.json`: all ablation counts, compression statistics, growth-order candidates, and checks.
- `manifest.json`: source archive hash, code hashes, input crop, coordinate conventions, output
  hashes, and completion status.

The four runs are `uniform_32`, `stride_20`, `uniform_32_center_51`, and `stride_20_center_51`.
The parent contains `validation.json`, `token_counts.png`, and the full suite's `tests.xml`.

Full validated 32-view region, writing a new timestamped directory:

```bash
PYTHONPATH=src venv/bin/python -m volumetric_reasoning.region_tokens \
  --source-volume outputs/stage_one_validation/uniform_32/volume.npz
```

The same central crop as the reported experiment (requested bounds snap outward to source faces):

```bash
PYTHONPATH=src venv/bin/python -m volumetric_reasoning.region_tokens \
  --source-volume outputs/stage_one_validation/uniform_32/volume.npz \
  --bounds 1.49 1.49 0.1 2.49 2.49 2.5 \
  --scene-token-budget 8192
```

`--scene-token-budget` only reports fit; it does not remove tokens or change the map. A supplied
`--output` must be new or empty. Legacy endpoint-occupancy archives are rejected; first convert
their evidence using the stage-one CLI.

Reproduce all four runs in a new directory, or audit the existing outputs:

```bash
PYTHONPATH=src venv/bin/python documents/validate_cuboids.py
PYTHONPATH=src venv/bin/python -m pytest -q
```

The script creates missing runs, verifies existing ones without overwriting them, and regenerates
the aggregate audit and figure. Use `--run-dir` for a separate complete reproduction.
