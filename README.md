# FreeSpace Tokens

FreeSpace Tokens tests whether explicit observed-free and unknown volumes help a vision-language
model answer spatial questions about indoor scenes. Posed ScanNet++ RGB-D views produce a
three-state grid using TSDF zero crossings and depth-ray visibility. A frozen DINOv2 encoder
supplies max-pooled surface features. Homogeneous free and unknown cells are packed separately
into cuboids. Qwen3-VL receives these tokens with world-coordinate 3D positions and is fine-tuned
with LoRA on the ScanNet++ subset of ViCA-322K.

- [Research proposal](documents/FreeSpace_Tokens_Research_Proposal.tex)
- [GPU runbook](documents/GPU_RUNBOOK.md) for download, preprocessing, training, and evaluation
- [Three-state map validation](documents/stage_one_validation.md)
- [Cuboid validation](documents/cuboid_validation.md)

On a Linux CUDA host, install the GPU environment from the repo root:

```bash
bash scripts/setup_gpu.sh
source .venv-gpu/bin/activate
```

The four matched variants are surface only, surface + free, surface + unknown, and surface + both.
The evaluation uses scenes held out from both training and validation. Geometry and small CPU
model fixtures have been exercised. Full pretrained CUDA execution and learned accuracy are
pending.

| Location | Purpose |
|---|---|
| `data_download/` | ScanNet++ downloader and token-free config template |
| `src/volumetric_reasoning/state_map.py` | TSDF fusion and three-state map export |
| `src/volumetric_reasoning/region_tokens.py` | Exact cuboid compression and token geometry |
| `src/volumetric_reasoning/features/` | Frozen encoder, backprojection, and feature pooling |
| `src/volumetric_reasoning/prototype/` | QA and scene preparation, training, inference, and comparison |
| `scripts/` | GPU setup, ROI demo, and ablation entry points |
| `tests/` | Geometry and data regression tests |

To run the CPU tests:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

See [dataset layout](data/README.md) for the raw RGB-D inputs. Datasets, feature caches, checkpoints,
and predictions are local artifacts.
