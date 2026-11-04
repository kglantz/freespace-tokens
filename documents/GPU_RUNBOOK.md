# FreeSpace Tokens prototype — GPU runbook

The audit exercised geometry on the local ScanNet++ scene and the pinned decoder on CPU with a tiny
randomly initialized Qwen3-VL checkpoint. All four variants passed forward/backward, cached versus
uncached decoding, and adapter reload checks. A tiny DINOv2 checkpoint processed all 32 real ROI
frames on CPU. Of 16,873 surface tokens, 14,327 received at least one feature assignment and 2,546
retained their zero visual vector. Full pretrained CUDA execution, GPU memory, and learned accuracy
remain unmeasured.

## What is implemented

The pipeline is RGB-D observations → three-state map → homogeneous cuboids → frozen image features
→ depth backprojection and surface association → channel-wise max pooling → learned visual/volume
projections and shared 3-D position features → Qwen3-VL decoder with world-coordinate 3-D MRoPE
→ answer-only LoRA training → checkpointed generation and four-ablation comparison.

- Frozen vision encoder: **facebook/dinov2-base**, patch size 14, default long image edge 1,008 px.
  The proposal leaves the pretrained vision encoder unspecified. Images are resized without crops
  or flips. Patch-to-depth alignment accounts for the actual resized patch-grid dimensions.
- Decoder: **Qwen/Qwen3-VL-4B-Instruct**, with its original pretrained text decoder and LM head.
  Qwen's own vision tower and DeepStack are unused; DINOv2 provides the proposal's visual features.
  Dense Qwen3-VL sizes can be selected with `--model`; MoE variants are not supported here.
- Trainable components: visual/volume projections, feature normalization, shared learned 3-D position
  MLP, three type embeddings, and rank-16 LoRA on decoder Q/K/V/O projections. Base decoder weights,
  input text embeddings, LM head, and vision encoder remain frozen.
- Surface features use channel-wise maximum pooling over assigned image features. A valid depth
  endpoint inside the indexed grid uses its containing occupied cell or the nearest occupied center
  within 5 cm. This association tolerance affects the feature assignment only.
- **Every surface cell and every selected cuboid is retained.** A surface cell without an assigned
  feature receives a zero visual vector while retaining position/type information. Cache metadata
  records these cases. No invented RGB evidence, feature-based token pruning, or silent truncation.
- All coordinates stay in the source world frame. Learned position inputs use metres. For rotary
  positions, x/y/z are divided by a fixed **0.02 m per rotary unit** for both token types. This is
  independent of cell size and cuboid size. Cuboid extents remain in metres in token content.
- Qwen3-VL's interleaved three rotary axes are supplied x/y/z instead of time/height/width. The
  separate first position row remains sequential for causal attention, including KV-cache decoding.
  Text uses scalar positions repeated on all three axes, with the same text offset across ablations.
- Scene tokens use deterministic lexicographic world-coordinate order. Cuboid extents encode volume;
  one cuboid contributes one embedding, not six coordinate text tokens.
- Answer-only logits avoid allocating scene-length × vocabulary logits. An unpadded training mask
  keeps the pinned decoder on the causal SDPA path when the backend supports it. Gradient
  checkpointing and mixed precision are enabled by default. Optional `torchrun` uses DDP with
  **one complete model replica per GPU**. Each GPU must fit the longest example it processes.

## 1. Machine and installation

Use Linux, Python 3.11, an NVIDIA CUDA GPU, enough host RAM for dense scene fusion, and local access
to the licensed ScanNet++ raw data. GPU memory requirements have not been measured. BF16 is the
default; use `--dtype float16` where BF16 is unavailable. Model and dataset downloads need network
access. Run all commands below from the repository root on the GPU machine.

Copy this checkout, including uncommitted new files. To reuse the existing region, also copy these
git-ignored outputs: `outputs/stage_one_validation/uniform_32/` and
`outputs/cuboid_validation/uniform_32/`. Copy the raw scene or mount its dataset on the GPU host.
Do not copy the macOS Python virtual environment to Linux.

```bash
bash scripts/setup_gpu.sh
source .venv-gpu/bin/activate
```

The script installs CUDA 12.8 PyTorch 2.8.0 wheels and Transformers **4.57.1**. Use a driver that
supports those wheels. If your machine uses the CUDA 12.6 wheels instead:

```bash
FST_TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash scripts/setup_gpu.sh
source .venv-gpu/bin/activate
```

`sdpa` is the default attention implementation and needs no separate FlashAttention build. To opt
into FlashAttention 2 on compatible hardware/toolchains:

```bash
python -m pip install packaging ninja
python -m pip install flash-attn --no-build-isolation
```

Then add `--attention flash_attention_2` to training or inference. Avoid unpinned Transformers
upgrades: the decoder integration deliberately targets its 4.57.1 position-ID interface.

## 2. Fast path: current 23,520-token region

The raw-data root must contain `data/1ada7a0617/iphone/{rgb.mkv,depth.bin,pose_intrinsic_imu.json}`.
For example, `--data-root /datasets/scannetpp` means the scene directory is
`/datasets/scannetpp/data/1ada7a0617/iphone`. In the existing local checkout the argument was `data`.

This command registers the existing map, checks its token count, extracts features, trains for 30
optimizer steps, and generates an answer. It retains the full 2.02 × 2.02 × 2.42 m region at 2 cm resolution:

```bash
bash scripts/run_current_roi.sh /datasets/scannetpp runs/current_roi
```

The included three QA records ask for the **known dimensions of this particular mapped ROI**.
They are explicitly a geometry-only debug dataset, not ViCA annotations, object-grounding labels,
held-out evaluation, or evidence of research performance. For project training use Section 4.

The individual commands, if you want to run each stage separately:

```bash
python -m volumetric_reasoning.prototype.prepare register \
  --index artifacts/current_roi/index.json --scene 1ada7a0617 --scope roi \
  --volume outputs/stage_one_validation/uniform_32/volume.npz \
  --regions outputs/cuboid_validation/uniform_32/token_geometry.npz

python -m volumetric_reasoning.features.extract \
  --index artifacts/current_roi/index.json --data-root /datasets/scannetpp

python -m volumetric_reasoning.prototype.train \
  --index artifacts/current_roi/index.json \
  --train-qa examples/current_roi_train.jsonl \
  --output runs/current_roi --variant surface_both \
  --epochs 10 --max-steps 30 --gradient-accumulation 1 --save-every 10

python -m volumetric_reasoning.prototype.infer \
  --checkpoint runs/current_roi/last \
  --index artifacts/current_roi/index.json --region-id 1ada7a0617 \
  --question 'What is the z-axis extent of the entire mapped region in metres? Answer with a number.'
```

Run the combined script **or** the individual commands. Registering the same key twice is rejected.
Feature caches require `--skip-existing` and identical map, calibration, encoder and extraction
settings. Caches from before the association fix need regeneration. Register a fresh index and
feature path using the individual commands if you need to keep an older cache. Training output
must be new unless `--resume` is specified. Choose another run directory for a new experiment.

## 3. Rebuild geometry if you did not transfer the outputs

This creates a fresh grid anchored at `(1,1,0.1)`. It has slightly different cell boundaries and
dimensions from the cached 23,520-token experiment, so its counts need not match. Register these
new paths and use QA appropriate to their bounds; the bundled ROI dimension questions refer to
the **cached** grid, not this fresh 2 × 2 × 2.4 m grid.

```bash
python -m volumetric_reasoning.state_map \
  --data-root /datasets/scannetpp --scene 1ada7a0617 --frames 32 \
  --bounds 1 1 0.1 3 3 2.5 --voxel-size 0.02 --truncation 0.08 \
  --free-margin 0.02 --skip-depth-check --output artifacts/fresh_roi/map

python -m volumetric_reasoning.region_tokens \
  --source-volume artifacts/fresh_roi/map/volume.npz \
  --output artifacts/fresh_roi/regions

python -m volumetric_reasoning.prototype.prepare register \
  --index artifacts/fresh_roi/index.json --scene 1ada7a0617 --scope roi \
  --volume artifacts/fresh_roi/map/volume.npz \
  --regions artifacts/fresh_roi/regions/token_geometry.npz

python -m volumetric_reasoning.features.extract \
  --index artifacts/fresh_roi/index.json --data-root /datasets/scannetpp
```

TSDF construction and cuboid construction currently run on CPU. Feature extraction, training, and
generation run on GPU. No mesh extraction is required.

## 4. Project training: ScanNet++ subset of ViCA-322K

Download the public QA annotations and install the ScanNet++ downloader dependencies:

```bash
python -m volumetric_reasoning.prototype.prepare download-qa --output data/vica
python -m pip install -r data_download/requirements.txt
```

If you need raw RGB-D data, use your [ScanNet++ access token](https://scannetpp.mlsg.cit.tum.de/scannetpp/).
The commands below choose 50 annotated scenes for a pilot and download their RGB video, depth, and
poses. The generated config is local to `data/` and prompts for the token.

```bash
export FST_DATA_ROOT="$PWD/data/scannetpp"
export FST_SCENE_COUNT=50
python - <<'PY'
import json
import os
from pathlib import Path, PurePosixPath
import random
import yaml

records = json.loads(Path('data/vica/scannetpp/base/room_size.json').read_text())
scenes = sorted({PurePosixPath(row['video']).stem for row in records})
selected = sorted(random.Random(42).sample(scenes, int(os.environ['FST_SCENE_COUNT'])))
Path('data/vica_scene_ids.txt').write_text('\n'.join(selected) + '\n')
config = yaml.safe_load(Path('data_download/download_scannetpp.example.yml').read_text())
config.update(data_root=os.environ['FST_DATA_ROOT'], download_scenes=selected)
Path('data/scannetpp_vica_download.yml').write_text(yaml.safe_dump(config))
PY
python data_download/download_scannetpp.py data/scannetpp_vica_download.yml
```

Use `$FST_DATA_ROOT` in place of `/datasets/scannetpp` in the remaining commands. If this root
contains additional scenes, pass `--scenes data/vica_scene_ids.txt` to the importer to keep the pilot cohort.

Import the ScanNet++ records, filter to raw scenes you have, and assign whole scenes to deterministic
80/10/10 train/validation/test splits. At least three scenes with both annotations and raw inputs
are required. The local checkout contains only `1ada7a0617`, a ScanNet++ validation scene. ViCA's
annotations come from the source datasets' training splits. Use the ROI demo with the local scene
and provision annotated ScanNet++ training scenes for this experiment.

```bash
python -m volumetric_reasoning.prototype.prepare import-vica \
  --input data/vica/scannetpp --data-root /datasets/scannetpp \
  --output data/qa --seed 42
```

Optional `--scenes path/to/scene_ids.txt` restricts the cohort. `--split-json` accepts your explicit
`{"train":[...],"val":[...],"test":[...]}` scene lists instead of random splits; they must be disjoint
and cover the imported scenes. The importer checks for RGB, depth, and pose files when given
`--data-root`. It excludes the appearance-order task because the static map discards video order.
Each answer in a multi-turn dialogue becomes its own example with preceding turns in `history`.
Training computes loss on the current answer. Evaluation supplies the reference history, so report
first-turn and later-turn scores separately when measuring independent scene grounding.
`--single-scene-debug` emits training data and empty held-out files for development.

Build geometry for every scene in the split manifest. Start with 32 selected views and 4 cm cells
for whole-scene QA. Use the 2 cm ROI demo above for the fine-resolution prototype.

```bash
python -m volumetric_reasoning.prototype.prepare build-scenes \
  --qa-dir data/qa --data-root /datasets/scannetpp \
  --output artifacts/scenes32 --frames 32 \
  --voxel-size 0.04 --truncation 0.08 --free-margin 0.02 --max-voxels 50000000
```

This path derives bounds from **all valid endpoints of the selected depth views plus 10 cm padding**.
It does not use the laser mesh or QA answer, and does not crop scene-wide annotations to the earlier
1 m or 2 m experiment. Coverage depends on what the selected views actually saw. The prototype
raises on oversized grids or sequences without dropping observations or tokens.

On `1ada7a0617`, the larger saved 32-view 2 cm map yielded **366,388** scene tokens, including
**293,889** surface tokens. Both exceed the configured 262,144-token context. A fresh 4 cm build
with bounds derived from the selected observations yielded **82,194** tokens. The two builds
used different bounds, so these counts illustrate capacity requirements rather than a controlled
resolution comparison. Full attention at 82,194 scene tokens still processes roughly 12 times as
many token pairs as at 23,520. Measure a single training step on the target GPU before starting
four runs.

Check training and validation geometry before extracting features:

```bash
python -m volumetric_reasoning.prototype.train \
  --index artifacts/scenes32/index.json \
  --train-qa data/qa/train.jsonl --val-qa data/qa/val.jsonl \
  --output runs/scenes32/surface_both --variant surface_both --preflight-only
```

Run the same check on test geometry by passing `data/qa/test.jsonl` as `--train-qa`, omitting
`--val-qa`, and retaining `--preflight-only`. Preflight reads geometry only. Allow additional room
for question, dialogue, answer, and generation tokens.

Cache frozen image features for all regions, once:

```bash
python -m volumetric_reasoning.features.extract \
  --index artifacts/scenes32/index.json --data-root /datasets/scannetpp --skip-existing
```

Or distribute independent scenes across four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m volumetric_reasoning.features.extract \
  --index artifacts/scenes32/index.json --data-root /datasets/scannetpp --skip-existing
```

Train the complete representation on one GPU:

```bash
python -m volumetric_reasoning.prototype.train \
  --index artifacts/scenes32/index.json \
  --train-qa data/qa/train.jsonl --val-qa data/qa/val.jsonl \
  --output runs/scenes32/surface_both --variant surface_both \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --epochs 1 --gradient-accumulation 8 --lr 0.0002 --save-every 100
```

For four GPUs, replace the initial `python` with `torchrun --standalone --nproc_per_node=4`.
Each process uses microbatch size 1; effective batch size is GPUs × gradient accumulation. For
equivalent batch size when changing GPU count, adjust accumulation accordingly. The code supports
single-node DDP; multi-node torchrun can supply its normal rendezvous settings.

For a one-update GPU pilot, use a fresh output directory, select a long example from the preflight
results, and set `--max-steps 1 --gradient-accumulation 1`. Omit `--val-qa` for the pilot or add
`--eval-max-examples 1`. The training log records sequence length, elapsed time, and peak allocated
GPU memory on rank zero.

With `--val-qa`, validation runs at epoch boundaries and at the final scheduled step. The default
uses a seeded sample of up to 128 validation records for checkpoint selection. Set
`--eval-max-examples 0` for all validation QA or `--eval-every N` for intermediate checks.

## 5. Four matched ablations

Use a new output root if you previously trained `surface_both` there. All four runs below use the
same observations, feature caches, QA, seed, model initialization, and training settings:

```bash
FST_GPUS=4 bash scripts/train_ablations.sh \
  artifacts/scenes32/index.json \
  data/qa/train.jsonl data/qa/val.jsonl \
  runs/ablations32 \
  --model Qwen/Qwen3-VL-4B-Instruct --epochs 1 --gradient-accumulation 2
```

Use `FST_GPUS=1` and accumulation 8 for the same effective batch size. The variants are `surface`,
`surface_free`, `surface_unknown`, and `surface_both`. Models are trained sequentially; each uses
the selected number of GPUs. All scene tokens belonging to a variant are kept. Text losses cover
answer tokens only. Model weights are frozen except the documented projections/embeddings/LoRA.

## 6. Resume

Resume a single run using the same number of GPUs, QA/index files, caches, accumulation, seed, and
intended total epoch count as the original run:

```bash
torchrun --standalone --nproc_per_node=4 -m volumetric_reasoning.prototype.train \
  --index artifacts/scenes32/index.json \
  --train-qa data/qa/train.jsonl --val-qa data/qa/val.jsonl \
  --output runs/ablations32/surface_both \
  --resume runs/ablations32/surface_both/last \
  --epochs 1 --gradient-accumulation 2
```

The checkpoint supplies the model, ablation, LoRA, and position settings. Optimizer, scaler, epoch,
and next batch are restored. Dropout RNG is reseeded on resume, so continuation is not claimed to
be bit-for-bit identical to an uninterrupted run. A save occurs at optimizer-step boundaries.
`--max-steps N` is a total step limit, including already completed steps, not N additional steps.

## 7. Inference and held-out comparison

Generate one answer from a trained checkpoint:

```bash
python -m volumetric_reasoning.prototype.infer \
  --checkpoint runs/ablations32/surface_both/best \
  --index artifacts/scenes32/index.json --region-id YOUR_SCENE_ID \
  --question 'Describe the observed free space around the furniture.' --max-new-tokens 256
```

Use `last` if no validation checkpoint was created. `YOUR_SCENE_ID` must be a registered index key.
Generation is greedy and uses a KV cache. An untrained/random projector is not useful for inference;
the CLI requires an adapter checkpoint created by training.

Evaluate every test question on all four variants:

```bash
bash scripts/evaluate_ablations.sh \
  artifacts/scenes32/index.json data/qa/test.jsonl \
  runs/ablations32 outputs/predictions32 --max-new-tokens 256
```

Or evaluate one checkpoint:

```bash
python -m volumetric_reasoning.prototype.infer \
  --checkpoint runs/ablations32/surface_both/best \
  --index artifacts/scenes32/index.json --qa data/qa/test.jsonl \
  --output outputs/predictions32/surface_both.jsonl --max-new-tokens 256
```

Do not run both examples into the same prediction file: existing files are preserved. Evaluation
checks training and validation scene provenance and rejects overlap unless `--allow-seen-scenes`
explicitly selects debug evaluation. `--limit N` limits the number of QA examples.

Prediction JSONL includes reference, generated answer, task, region, model variant, and input-token
count. Metric JSON includes normalized exact match and relative accuracy for bare scalar answers
at 5/10/20/25%, overall and by task. Comparison checks QA identity, including dialogue history,
and available provenance for training schedules, model settings, and input caches. For a paper,
use task-specific scoring for units, multiple-choice answers, and semantically equivalent free text.
The ViCA authors report their benchmark results on VSI-Bench, a separate evaluation dataset.

## 8. Your own ROI-specific QA

Each line of a QA file is:

```json
{"id":"unique_id","scene_id":"1ada7a0617","region_id":"1ada7a0617","scope":"roi","question":"Your question about this mapped region","answer":"Your ground-truth answer","task":"optional_task_name"}
```

`scope: "roi"` is required for explicitly cropped entries. Global ViCA annotations use `scope:
"scene"` and are rejected against an ROI entry. This prevents silently training room-size/object-count
answers against an arbitrary crop. It does not automatically establish that custom labels are correct.
The scene index stores portable paths relative to itself; move it together with the referenced files.

For additional view-count conditions, build into separate roots (e.g. `--frames 276 --output
artifacts/scenes276`), extract separate caches, and train all four variants with the same frame
selection within that condition. This selects 276 uniformly spaced views; it is not the old
stride-20 comparison unless you register that exact existing map instead.

## Artifacts and implementation files

- Geometry: existing `state_map` and `region_tokens` CLIs.
- Features: `features/extract.py`; `surface_features.npz` plus cache provenance/coverage JSON.
- Dataset preparation: `prototype/prepare.py`; QA JSONL, split manifest, and scene index.
- Model: `prototype/model.py`; visual/volume heads, shared position encoding, native 3-axis MRoPE,
  explicit low-rank adapters, answer-only loss, and autoregressive KV-cache generation.
- Training: `prototype/train.py`; run manifest, JSONL logs, `last/` and optional `best/` checkpoints.
- Inference/evaluation: `prototype/infer.py`, `prototype/compare.py`.
- Checkpoint: `model_config.json`, `adapter.pt`, `training.pt`, `data_manifest.json`. Frozen base
  weights are not duplicated; the resolved model revision is recorded and loaded from Hugging Face.

## Primary references used for implementation

- [Proposal](FreeSpace_Tokens_Research_Proposal.tex).
- [Pinned Qwen3-VL decoder implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py).
- [DINOv2 model and preprocessing](https://huggingface.co/facebook/dinov2-base).
- [ViCA-322K annotations](https://huggingface.co/datasets/nkkbr/ViCA-322K/tree/main/scannetpp).
- [ViCA paper and evaluation protocol](https://arxiv.org/abs/2505.12312).
- [Official PyTorch CUDA wheel versions](https://pytorch.org/get-started/previous-versions/).
