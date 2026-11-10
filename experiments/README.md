# Experiments

Notebooks for analysing the block-map pipeline (`scripts/step1_reconstruction.py` … `step3_tokens.py`).
They load the files the scripts write to `outputs/<scene>/` and reuse the code in `src/volumetric_reasoning/`;
they do not define their own pipeline.

Please **clear outputs before committing** notebooks (embedded figures bloat the repo).

| Notebook | Purpose |
|---|---|
| `10_free_space_boxes.ipynb` | How free / unknown blocks are merged into box tokens: step-by-step box growth, box size / shape statistics, dependence on growth order, size cap and block size; which surface blocks get image features |

`old/` holds the notebooks of the earlier dense-TSDF pipeline (see `old/README.md`).
