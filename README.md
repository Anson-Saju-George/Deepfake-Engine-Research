![DeepFake Detection Research Pipeline Banner](images/Banner.png)

# DeepFake Detection Research Pipeline

> Research-grade image and video deepfake detection pipeline with protocol-aware loading, dataset-audit tooling, and reproducible experiment tracking.


## Overview

This repository is a research-focused deepfake detection project built around one core idea: do not treat images, videos, and derived frames as the same problem by default.

The pipeline is designed to:

- keep image and video protocols separate by default
- preserve defensible dataset boundaries
- validate the dataset state directly from disk
- support spatial and spatial+temporal video learning cleanly
- maintain paper-ready experiment records instead of ad hoc runs

The result is a cleaner methodology for both research reporting and practical engineering review.

## Why This Repo Matters

Most deepfake repos collapse everything into one loader and one vague binary task. This project does the opposite.

The current pipeline explicitly distinguishes:

- `image_only` for image-domain spatial learning
- `video_only` with `mode="single"` for video-domain spatial baselines
- `video_only` with `mode="sequence"` for spatial+temporal raw-video learning
- `frame_only` for optional derived frame-folder experiments
- `combined_aux` only for auxiliary mixed-media studies

That separation reduces methodological sloppiness, helps prevent accidental split leakage, and makes the experiments easier to defend in a paper or technical interview.

## Dataset Snapshot

Current cleaned raw dataset state:

- total raw samples: `192,016`
- images: `179,988`
- videos: `12,028`
- labels: `99,740 fake`, `92,276 real`

Image datasets:

- `cifake`
- `ai-generated-images-vs-real-images`

Video datasets:

- `celeb-df-v2`
- `faceforensics++`
- `real-ai-videos`

Historical cleaning notes:

- `12` corrupted images were deleted from `ai-generated-images-vs-real-images`
- `3` bad videos were removed earlier due to old `moov`-type issues

## The Dataloader Pipeline

The dataloader is the core of the project.

What it does well:

- scans raw images and raw videos from a unified entrypoint
- resolves protocol automatically from media type when appropriate
- preserves source image `train/test` boundaries
- derives image validation only from source `train`
- performs identity-aware `70/10/20` splitting for videos
- supports contiguous clip sampling for sequence learning
- uses random contiguous clips for train
- uses center contiguous clips for validation and test
- supports optional derived frame folders without making them the default path

This is the main engineering strength of the repo.

It is not just “a loader.” It is a protocol-aware experimental control layer:

- correct for benchmark-style image datasets
- correct for identity-sensitive video datasets
- flexible enough for frame-derived ablations
- explicit enough to explain in a research paper

![DeepFake Detection Research Pipeline Banner](images/Deepfake_detection_research_pipeline_overview.png)

## Research Workflow

The repo is organized around three layers:

### 1. Canonical Data Layer

- `data/dataloader.py`
- `data/dataset.md`
- `data/commands.md`

This layer defines dataset truth, loading semantics, split policy, and audit utilities.

### 2. Research Knowledge Base

- `books/`

This contains lifecycle-oriented methodology notes:

- problem definition
- data collection
- cleaning and preprocessing
- EDA
- model selection
- training
- evaluation
- tuning
- maintenance

### 3. Training Layer

- `train/`

Current implemented state:

- image smoke validation is active
- ViT image training is active
- video planning is documented
- new-tree video execution is still pending

## Current Training State

Implemented image commands:

```bash
python -m train.image_simulate
```

```bash
python -m train.run_image_vit
```

Current image trainer coverage:

- `IMG-EXP-01` -> `vit_base_patch16_224`
- `IMG-EXP-02` -> `vit_large_patch16_224`
- `IMG-EXP-03` -> `vit_huge_patch14_224`

Current implemented save layout:

```text
train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/
```

Per-run artifacts include:

- `best.pth`
- `last.pth`
- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`

## Tech Stack

- Python
- PyTorch
- Torchvision
- timm
- NumPy
- scikit-learn
- tqdm
- Pillow
- OpenCV
- decord

## Repo Structure

```text
books/       Research lifecycle notes and methodology records
data/        Dataloader, dataset audit tools, and canonical dataset docs
datasets/    Raw datasets and optional derived frame roots
proc/        Optional preprocessing utilities
train/       New training tree and training command docs
train_old/   Legacy training reference tree
```

## Recommended First Commands

Audit dataset truth:

```bash
python -m data.dataset_analyzer
```

Validate image/video readability:

```bash
python -m data.dataset_run --dtype image --num-workers 16
python -m data.dataset_run --dtype video --num-workers 8
```

Smoke-test the image pipeline:

```bash
python -m train.image_simulate
```

Run the first real image baseline:

```bash
python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope image_combined
```

## Current Position

This repo is already strong on:

- dataset auditing
- split correctness
- protocol separation
- research documentation
- reproducible run records

The next major step is expanding the new training tree beyond ViT image runs and migrating the video execution path fully into the same clean framework.
