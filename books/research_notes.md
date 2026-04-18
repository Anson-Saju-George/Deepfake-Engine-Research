# Research Notes

## Scope

This file captures repo-verified research truth, major design decisions, known caveats, and the practical interpretation of the current codebase.

Primary anchors:

- `data/dataloader.py`
- `data/dataset.md`
- `data/commands.md`
- `proc/pre_process_videos.py`
- `train/image/`
- `train/video/`
- `train_old/` for historical comparison

## Current Repo Truth

### Active data layer

- active dataloader: `data/dataloader.py`
- active image protocol: `image_only`
- active video protocol: `video_only`
- active frame protocol: `frame_only`
- auxiliary mixed-media protocol: `combined_aux`

Operational meaning:

- image-only experiments are the image-domain spatial benchmark family
- video-only `single` mode is the video-domain spatial baseline
- video-only `sequence` mode is the clip-based spatial+temporal path
- raw-video loading is the intended main video path
- derived frame folders remain auxiliary only

### Active image training tree

The image tree has been reorganized under `train/image/`.

Current active paths:

- smoke test: `python -m train.image.simulate_image_train`
- runners:
  - `python -m train.image.run_image_vit`
  - `python -m train.image.run_image_convnext`
  - `python -m train.image.run_image_swin`
  - `python -m train.image.run_image_deit`
  - `python -m train.image.run_image_convnextv2`
  - `python -m train.image.run_image_maxvit`
  - `python -m train.image.run_image_eva`
- active registry: `train/image/image_models.py`
- compatibility shim retained: `train/image/image_model_configs.py`

Important historical change:

- EfficientNet and ResNet were removed from the active image plan because they no longer met the current model-selection requirements
- the active image families are now:
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

### Active video research tree

The video tree has been reorganized into category packages:

- `train/video/spa/`
- `train/video/tmp/`
- `train/video/st/`

Active video registry source of truth:

- `train/video/video_models.py`

Current video execution truth:

- category-specific runners are active
- resolved configs, dataset scopes, and save paths are active
- current category runners execute a real timm-backed video trainer for image-style video backbones
- the separate raw-video smoke path still exists for pipeline validation
- reserved native-video IDs `VID-ST-07..12` are not active in the current runner surface
- the raw-video smoke path supports optional FFmpeg hardware decode backends, OpenCV decode, bad-video auditing, and timing instrumentation

Active video runners:

- `python -m train.video.spa.run_video_spatial`
- `python -m train.video.tmp.run_video_temporal`
- `python -m train.video.st.run_video_spatiotemporal`
- `python -m train.video.simulate_video_train`

Video smoke decode note:

- explicit decode backends now include `cv2`, `ffmpeg`, `ffmpeg_qsv`, `ffmpeg_d3d11va`, and opt-in `decord`
- current stable default for the active runner surface is `cv2`
- Intel GPU decode remains available through `--decode-backend ffmpeg_qsv`
- `decord` is no longer treated as the default because corrupted H.264 samples can crash workers
- this changes the decode backend, not the benchmark definition or split policy

Current stable smoke runtime defaults:

- decode backend: `cv2`
- transform profile: `smoke_fast`
- GPU augmentation: `none`
- workers: `8`
- prefetch factor: `4`

Current active imbalance-handling variants in the video runner surface:

- `none`
  - standard cross-entropy
  - train-only oversampling remains enabled
- `weighted_ce`
  - train oversampling disabled
  - class-weighted cross-entropy from the natural train split
- `focal`
  - train oversampling disabled
  - focal loss with class-weighted alpha from the natural train split

Compatibility alias:

- `weighted_bce` is accepted at the CLI level but normalizes to `weighted_ce`
- the active timm-backed video trainer still uses a 2-logit softmax head, so BCE is not the internal criterion form

Recent smoke/runtime additions:

- `python -m train.video.simulate_video_train_base`
- `python -m train.video.simulate_video_train_kornia`
- `python -m train.video.bad_video_test`
- per-split timing output:
  - batch wait time
  - compute time
  - average wait per batch
  - average compute per batch

## Dataset Snapshot

Latest cleaned raw dataset state:

- total raw samples: `192,016`
- images: `179,988`
- videos: `12,028`
- labels: `99,740 fake`, `92,276 real`

Historical cleanup notes:

- `12` corrupted images were deleted from `ai-generated-images-vs-real-images`
- `3` bad videos had already been deleted earlier due to old `moov`-type issues

Important note:

- extracted frame folders can increase ad hoc dataloader-visible totals when frame roots exist on disk
- `data.dataset_analyzer` intentionally reports the raw image and raw video corpora used by the main training path

## Split And Sampling Policy

### Images

- preserve original source dataset train and test boundaries
- derive validation from source training data
- do not globally reshuffle image datasets into a pooled split

### Videos

- use identity-aware grouping for split assignment
- default target split is `70/10/20`
- balancing is applied only after split construction
- balancing is restricted to the training split
- validation and test remain untouched
- `single` is the spatial baseline on videos
- `sequence` uses contiguous clips for clip-based learning
- train clip sampling is random contiguous
- evaluation clip sampling is center contiguous

Methodological interpretation:

- the repo does not use naive random file-level video splitting
- the repo does not rebalance validation or test for video evaluation
- split defensibility is treated as the first constraint
- train-only oversampling is treated as the second constraint

### Frame folders

- treat extracted frame folders as derived-video data, not a native benchmark
- use identity-aware splitting when they are used

## Current Image Optimization Policy

The active image families currently share one optimization stack by design.

Shared across:

- ViT
- ConvNeXt
- Swin
- DeiT
- ConvNeXtV2
- MaxViT
- EVA

Current shared optimizer and control stack:

- optimizer: `AdamW`
- base LR: `1e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR`
- minimum LR: `1e-6`
- warmup: `1` epoch
- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- EMA decay: `0.999`
- gradient clipping: `1.0`
- early stopping patience: `3`
- minimum improvement delta: `1e-4`
- best-checkpoint metric: `val_f1`
- mixed precision on CUDA

Current baseline defaults:

- epochs: `10`
- batch size: `64`

## Current Video Registry Ordering Truth

The video registry is now ordered strictly by:

- category first
- then paradigm
- then family
- then parameter scale

Spatial ordered ladder:

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

Temporal ordered ladder:

- `VID-TMP-01` ConvNeXt-Base sequence
- `VID-TMP-02` ConvNeXt-Large sequence
- `VID-TMP-03` ConvNeXtV2-Base sequence
- `VID-TMP-04` Swin-Base sequence
- `VID-TMP-05` Swin-Large sequence
- `VID-TMP-06` MaxViT-Base sequence

Spatiotemporal ordered ladder:

- `VID-ST-01` Xception71 hybrid
- `VID-ST-02` ConvNeXt-Base hybrid
- `VID-ST-03` ConvNeXt-Large hybrid
- `VID-ST-04` Swin-Base hybrid
- `VID-ST-05` MaxViT-Base hybrid
- `VID-ST-06` MaxViT-Large hybrid
- `VID-ST-07..12` are reserved native-video vacancies and are not active in the current timm-backed runner surface

## Current Experiment Record Truth

Current visible saved image-run artifacts under `train/image/` include:

- ViT:
  - `IMG-EXP-01_vit_base_patch16_224_image_combined`
  - `IMG-EXP-02_vit_large_patch16_224_image_combined`
- ConvNeXt:
  - `IMG-EXP-04_convnext_base_image_combined`
  - `IMG-EXP-05_convnext_large_image_combined`
- Swin:
  - `IMG-EXP-07_swin_base_patch4_window7_224_image_combined`

Current visible saved video-run artifacts under `train/video/` include:

- spatial:
  - `VID-SPA-02_convnext_base_video_combined_loss-none`
  - `VID-SPA-02_convnext_base_video_combined_loss-weighted_ce`
  - `VID-SPA-02_convnext_base_video_combined_loss-focal`
  - `VID-SPA-06_swin_base_patch4_window7_224_video_combined_loss-none`
- temporal:
  - `VID-TMP-01_convnext_base_video_combined_loss-none`
  - `VID-TMP-01_convnext_base_video_combined_loss-weighted_ce`
  - `VID-TMP-02_convnext_large_video_combined_loss-none_lr-5e-05`
- spatiotemporal:
  - `VID-ST-02_convnext_base_video_combined_loss-none`
  - `VID-ST-03_convnext_large_video_combined_loss-none_lr-5e-05`
  - `VID-ST-05_maxvit_base_tf_224.in1k_video_combined_loss-none_lr-5e-05`

## Current Saved Result Snapshot

### Image Results

Current completed image runs with saved final summaries:

- `IMG-EXP-01` ViT-Base
  - best val F1: `0.9734`
  - test F1: `0.9702`
  - test accuracy: `0.9703`
- `IMG-EXP-02` ViT-Large
  - best val F1: `0.9592`
  - test F1: `0.9546`
  - test accuracy: `0.9548`
- `IMG-EXP-04` ConvNeXt-Base
  - best val F1: `0.9869`
  - test F1: `0.9863`
  - test accuracy: `0.9863`
- `IMG-EXP-05` ConvNeXt-Large
  - best val F1: `0.9864`
  - test F1: `0.9840`
  - test accuracy: `0.9840`
- `IMG-EXP-07` Swin-Base
  - best val F1: `0.9861`
  - test F1: `0.9842`
  - test accuracy: `0.9842`

Current image leader:

- `IMG-EXP-04` ConvNeXt-Base

### Video Results

Current completed video runs with saved final summaries:

- spatial
  - `VID-SPA-02` ConvNeXt-Base, `loss=none`
    - best val F1: `0.7246`
    - test F1: `0.7023`
    - test accuracy: `0.8566`
  - `VID-SPA-02` ConvNeXt-Base, `loss=weighted_ce`
    - best val F1: `0.7161`
    - test F1: `0.6709`
    - test accuracy: `0.8717`
  - `VID-SPA-02` ConvNeXt-Base, `loss=focal`
    - best val F1: `0.5312`
    - test F1: `0.4744`
    - test accuracy: `0.8027`
  - `VID-SPA-06` Swin-Base, `loss=none`
    - best val F1: `0.5810`
    - test F1: `0.5253`
    - test accuracy: `0.7504`
- temporal
  - `VID-TMP-01` ConvNeXt-Base, `loss=none`
    - best val F1: `0.6998`
    - test F1: `0.7090`
    - test accuracy: `0.8652`
  - `VID-TMP-01` ConvNeXt-Base, `loss=weighted_ce`
    - best val F1: `0.7339`
    - test F1: `0.7099`
    - test accuracy: `0.8832`
  - `VID-TMP-02` ConvNeXt-Large, `loss=none`, `seq_len=4`, `lr=5e-5`
    - best val F1: `0.7752`
    - test F1: `0.7841`
    - test accuracy: `0.9089`
- spatiotemporal
  - `VID-ST-02` ConvNeXt-Base hybrid, `loss=none`
    - best val F1: `0.5977`
    - test F1: `0.5744`
    - test accuracy: `0.7663`
  - `VID-ST-03` ConvNeXt-Large hybrid, `loss=none`, `seq_len=4`, `lr=5e-5`
    - best val F1: `0.7752`
    - test F1: `0.7841`
    - test accuracy: `0.9089`
  - `VID-ST-05` MaxViT-Base hybrid, `loss=none`, `seq_len=4`, `lr=5e-5`
    - best val F1: `0.6848`
    - test F1: `0.6414`
    - test accuracy: `0.8186`

Current video leaders:

- spatial leader: `VID-SPA-02` ConvNeXt-Base, `loss=none`
- temporal leader: `VID-TMP-02` ConvNeXt-Large
- spatiotemporal leader: `VID-ST-03` ConvNeXt-Large hybrid

Important implementation caveat:

- the current `temporal` and `spatiotemporal` categories both use the same active sequence trainer class
- the current sequence path performs per-frame image-backbone encoding followed by temporal mean pooling
- therefore `VID-TMP-02` and `VID-ST-03` should not be reported as fundamentally different native-video architectures
- the current `ST` branch is best described as hybrid clip aggregation under the same active sequence implementation

## Consolidated Evidence-Based Interpretation

The current completed evidence supports the following thesis-level interpretation.

### Image story

- image detection is presently the easiest benchmark family in the repository
- the strongest completed image result is already near-saturated relative to the current datasets
- ConvNeXt-Base is the strongest saved image result
- Swin-Base is competitive but not ahead
- the completed ViT runs are clearly weaker than the strongest CNN and hierarchical-transformer image runs

### Video story

- spatial video baselines are useful, but clearly weaker than the best clip-based result
- the strongest completed raw-video result is `VID-TMP-02` ConvNeXt-Large sequence
- the current completed evidence favors ConvNeXt over the finished Swin and MaxViT video runs
- more complex imbalance handling did not clearly beat the train-only oversampling baseline

### Cross-domain story

- image leader test F1: `0.9863`
- video leader test F1: `0.7841`
- this gap is one of the most important research findings currently available in the repo
- it shows that raw-video deepfake detection remains much harder than image-domain deepfake detection under the present methodology

### Taxonomy story

- the registry still contains spatial, temporal, and spatiotemporal categories
- the current validated implementation truth is closer to:
  - spatial modeling
  - clip-based modeling
- this is because active `TMP` and `ST` runs both use per-frame image encoding followed by temporal mean pooling

## Practical Problems Encountered And How They Were Handled

The final documentation should preserve the actual engineering problems faced during experimentation.

### Problem: video decode and input delivery became the main bottleneck

Observed symptoms:

- CPU-heavy sequence runs
- low GPU utilization during video training
- long epoch times even when model compute itself was small

What changed:

- input-path instrumentation was added through the smoke runner
- worker count, prefetching, reader caching, and lighter smoke transforms were tested
- RAM caching was added:
  - full train/val/test caching for `single` mode
  - val/test caching only for `sequence` mode

What survived:

- caching is useful
- but the train split for sequence mode still streams, so raw-video input remains the dominant cost

### Problem: FFmpeg hardware decode was not a stable default on the real corpus

Observed symptoms:

- repeated `ffmpeg_qsv` decode failures
- large bad-video audit logs
- no decisive practical throughput win on the current machine and corpus

Evidence anchor:

- `train/video/bad_video_log.jsonl`

What changed:

- hardware decode support was kept as an option rather than treated as the default path
- the stable default category-runner runtime stayed on `cv2`

### Problem: Decord was promising but unsafe as the default

Observed symptoms:

- worker exits on damaged H.264 samples
- instability on mixed-quality raw-video data

What changed:

- Decord was retained as an opt-in decode backend
- it was not adopted as the default research path

### Problem: GPU augmentation did not solve the main bottleneck by itself

Observed symptoms:

- Kornia moved augmentation to CUDA, but decode and input wait remained the dominant bottleneck

What changed:

- Kornia support was retained as an option
- `gpu_aug = none` remained the stable default for the active runner surface

### Problem: the original spatiotemporal framing overstated implementation diversity

Observed symptoms:

- `VID-TMP-02` and `VID-ST-03` converged to the same saved result
- code inspection showed the same active sequence aggregation mechanics underneath both categories

What changed:

- docs now preserve the registry categories
- but the research interpretation explicitly states that current validated `TMP` and `ST` evidence collapses to one clip-based aggregation framework

Observed documentation risk:

- historical run folders are not perfectly uniform in artifact completeness
- reporting should cite the files actually present in each run directory instead of assuming every saved folder has the full artifact set

## Current Source Of Truth Docs

Data layer:

- `data/dataset.md`
- `data/commands.md`

Image training docs:

- `train/image/image_model_config.md`
- `train/image/image_commands.md`

Video training docs:

- `train/video/video_config.md`
- `train/video/video_commands.md`
- `train/video/Experiment_List.md`

## Reporting Guidance

The thesis or paper should state clearly that:

- image and video experiments are separate protocol families
- raw-video training is the intended main video path
- frame-only experiments are auxiliary and derived from raw videos
- the active image tree was migrated into `train/image/`
- the active video research tree was reorganized into `spa`, `tmp`, and `st`
- the current video registry and runner surface are active and execute a real timm-backed trainer for image-style backbones
- reserved native-video IDs `VID-ST-07..12` are intentionally inactive in the current runner surface
