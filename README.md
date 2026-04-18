![DeepFake Detection Research Pipeline Banner](images/Banner.png)

# DeepFake Detection Research Pipeline

> Research-grade image and video deepfake detection pipeline with protocol-aware loading, dataset-audit tooling, raw-video-first methodology, and reproducible experiment tracking.

## Overview

This repository is built around one central rule: image datasets, raw-video datasets, and derived frame folders should not be treated as the same experimental object by default.

The current project state is organized around:

- a protocol-aware dataloader in `data/dataloader.py`
- an active image training tree under `train/image/`
- an active video research registry under `train/video/`
- thesis-oriented research notes under `books/`

That separation is methodological, not cosmetic. It exists to keep the benchmark claims defensible in a thesis, journal paper, or technical audit.

## Dataset Snapshot

Current cleaned raw dataset truth:

- total raw samples: `192,016`
- images: `179,988`
- videos: `12,028`
- labels: `99,740 fake`, `92,276 real`

Primary image datasets:

- `cifake`
- `ai-generated-images-vs-real-images`

Primary raw-video datasets:

- `celeb-df-v2`
- `faceforensics++`
- `real-ai-videos`

Historical cleanup notes:

- `12` corrupted images were deleted from `ai-generated-images-vs-real-images`
- `3` bad videos had already been removed historically due to old `moov`-type issues

## Dataloader Truth

The active dataloader is `data/dataloader.py`.

Supported protocol meanings:

- `image_only`: image-domain spatial learning
- `video_only`: raw-video learning
- `frame_only`: derived-frame experiments
- `combined_aux`: auxiliary mixed-media studies

Resolution rules:

- `dtype="image"` resolves to image-only behavior
- `dtype="video"` resolves to video-only behavior
- `dtype="frame"` resolves to frame-only behavior

Video sampling semantics:

- `mode="single"`: one sampled frame from a raw video, used as the video-domain spatial baseline
- `mode="sequence"`: contiguous ordered clip from a raw video
- train clip sampling: random contiguous
- eval clip sampling: center contiguous

Split rules:

- image experiments preserve original source dataset boundaries
- image validation is derived from source training data
- video and frame experiments use identity-aware grouped splitting

## Current Training Structure

### Active Image Tree

The active image tree lives under `train/image/`.

Current image command sheet:

- `train/image/image_commands.md`

Current image registry source of truth:

- `train/image/image_models.py`

Compatibility shims retained:

- `train/image/image_model_configs.py`

Active image families:

- ViT
- ConvNeXt
- Swin
- DeiT
- ConvNeXtV2
- MaxViT
- EVA

Active image experiment ladder:

- `IMG-EXP-01..03`: ViT
- `IMG-EXP-04..06`: ConvNeXt
- `IMG-EXP-07..08`: Swin
- `IMG-EXP-09`: DeiT
- `IMG-EXP-10..11`: ConvNeXtV2
- `IMG-EXP-12`: MaxViT
- `IMG-EXP-13..14`: EVA

Active image module paths:

```bash
python -m train.image.simulate_image_train
python -m train.image.run_image_vit
python -m train.image.run_image_convnext
python -m train.image.run_image_swin
python -m train.image.run_image_deit
python -m train.image.run_image_convnextv2
python -m train.image.run_image_maxvit
python -m train.image.run_image_eva
```

Current image save layout:

```text
train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/
```

### Active Video Research Tree

The active video tree lives under `train/video/`.

Current video command sheet:

- `train/video/video_commands.md`

Current video registry source of truth:

- `train/video/video_models.py`

Current video category packages:

- `train/video/spa/`
- `train/video/tmp/`
- `train/video/st/`

Current executable video modules:

```bash
python -m train.video.simulate_video_train
python -m train.video.simulate_video_train_base
python -m train.video.simulate_video_train_kornia
python -m train.video.bad_video_test
python -m train.video.spa.run_video_spatial
python -m train.video.tmp.run_video_temporal
python -m train.video.st.run_video_spatiotemporal
```

Important current truth:

- experiment resolution and runner layout are active
- category-specific video runners now execute a real timm-backed trainer for image-style video backbones over the raw-video loader path
- the separate smoke utility still exists for pipeline validation and decode benchmarking
- native video backbones such as Video Swin, TimeSformer, and MViT remain a later extension area
- the smoke path now supports:
  - OpenCV decode
  - optional FFmpeg-based hardware decode backends, including Intel Quick Sync via `--decode-backend ffmpeg_qsv`
  - optional Kornia GPU augmentation
  - alternate imbalance-handling loss variants
  - bad-video auditing and logging
  - per-split input-vs-compute timing
- current stable runner default is the `cv2` decode path rather than `ffmpeg_qsv` or `decord`

## Video Registry State

The video registry is now organized into three independent categories.

### Spatial

Ordered by paradigm, family, and parameter scale:

- `VID-SPA-01` Xception71
- `VID-SPA-02` ConvNeXt-Base
- `VID-SPA-03` ConvNeXt-Large
- `VID-SPA-04` ConvNeXtV2-Base
- `VID-SPA-05` ConvNeXtV2-Large
- `VID-SPA-06` Swin-Base
- `VID-SPA-07` Swin-Large
- `VID-SPA-08` ViT-Base
- `VID-SPA-09` EVA-Base
- `VID-SPA-10` MaxViT-Base
- `VID-SPA-11` MaxViT-Large

### Temporal

- `VID-TMP-01` ConvNeXt-Base sequence
- `VID-TMP-02` ConvNeXt-Large sequence
- `VID-TMP-03` ConvNeXtV2-Base sequence
- `VID-TMP-04` Swin-Base sequence
- `VID-TMP-05` Swin-Large sequence
- `VID-TMP-06` MaxViT-Base sequence

### Spatiotemporal

- `VID-ST-01` Xception71 hybrid
- `VID-ST-02` ConvNeXt-Base hybrid
- `VID-ST-03` ConvNeXt-Large hybrid
- `VID-ST-04` Swin-Base hybrid
- `VID-ST-05` MaxViT-Base hybrid
- `VID-ST-06` MaxViT-Large hybrid
- `VID-ST-07..12` are reserved native-video vacancies and are not active in the current timm-backed runner surface

## Shared Image Optimization Truth

The active image trainers currently share one optimization stack by design:

- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- base LR: `1e-4`
- weight decay: `1e-4`
- min LR: `1e-6`
- warmup: `1` epoch
- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- EMA decay: `0.999`
- gradient clipping: `1.0`
- best checkpoint metric: `val_f1`
- early stopping patience: `3`
- mixed precision on CUDA

Current image baseline defaults:

- epochs: `10`
- batch size: `64`

## Documentation Map

Canonical data and audit docs:

- `data/dataset.md`
- `data/commands.md`

Training docs:

- `train/image/image_commands.md`
- `train/image/image_model_config.md`
- `train/video/video_commands.md`
- `train/video/video_config.md`
- `train/video/Experiment_List.md`

Research lifecycle docs:

- `books/README.md`
- `books/research_notes.md`
- `books/experiment_matrix.md`
- `books/06_model_selection.md`
- `books/07_model_training.md`
- `books/09_hyperparameter_tuning.md`

## Recommended First Commands

Audit dataset truth:

```bash
python -m data.dataset_analyzer
```

Validate image and raw-video readability:

```bash
python -m data.dataset_run --dtype image --num-workers 16
python -m data.dataset_run --dtype video --num-workers 8
```

Smoke-test the image path:

```bash
python -m train.image.simulate_image_train
```

Smoke-test the raw-video path:

```bash
python -m train.video.simulate_video_train
```

Run the first image baseline:

```bash
python -m train.image.run_image_vit --exp IMG-EXP-01 --dataset-scope image_combined
```

Resolve the first video baseline:

```bash
python -m train.video.spa.run_video_spatial --exp VID-SPA-01 --dataset-scope video_combined
```
