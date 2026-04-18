# Model Training

## What This Stage Means

Model training is where cleaned, split, and protocol-correct data is turned into learned decision rules.

In this repository, training is not treated as an isolated script. It is the stage where all prior methodological choices become operational:

- protocol separation
- split defensibility
- sampling logic
- optimization controls
- checkpoint rules
- run-record preservation

## Current Training Tree Truth

### Active image training tree

The active image implementation now lives under `train/image/`.

Active image files include:

- `train/image/image_models.py`
- `train/image/image_model_config.md`
- `train/image/image_commands.md`
- `train/image/simulate_image_train.py`
- `train/image/run_image_vit.py`
- `train/image/run_image_convnext.py`
- `train/image/run_image_swin.py`
- `train/image/run_image_deit.py`
- `train/image/run_image_convnextv2.py`
- `train/image/run_image_maxvit.py`
- `train/image/run_image_eva.py`

### Active video research tree

The active video implementation and registry layer now lives under `train/video/`.

Active video files include:

- `train/video/video_models.py`
- `train/video/video_commands.md`
- `train/video/video_config.md`
- `train/video/Experiment_List.md`
- `train/video/simulate_video_train.py`
- `train/video/spa/run_video_spatial.py`
- `train/video/tmp/run_video_temporal.py`
- `train/video/st/run_video_spatiotemporal.py`

Important truth:

- video experiment resolution and command surfaces are active
- current category runners now execute a real timm-backed video trainer for image-style video backbones
- the separate smoke utility still exists for pipeline validation
- true video backbone support still remains a later extension area

### Legacy reference tree


## Protocol-Aware Training Philosophy

### Image training

Image training uses:

- `protocol = image_only`

Meaning:

- image datasets are treated as image-domain spatial tasks
- source dataset boundaries are preserved
- this is the clean benchmark path for image-only research claims

### Video training

Video training uses:

- `protocol = video_only`

Meaning:

- raw videos are the source of truth
- `mode="single"` is the video-domain spatial baseline
- `mode="sequence"` is the contiguous clip path for temporal or spatiotemporal work

### Frame-derived training

Frame-derived training uses:

- `protocol = frame_only`

Meaning:

- frame folders are derived-video artifacts
- they should be labeled as such in reporting

### Mixed auxiliary training

`combined_aux` exists, but it is not the core benchmark story.

## Split Logic And Why It Matters

### Images

- preserve source train/test boundaries
- derive validation from source training data
- do not reshuffle the whole image corpus into a pooled split

### Videos

- use identity-aware `70/10/20` splitting
- avoid subject leakage across train, validation, and test
- apply class balancing only after the identity-aware split is fixed
- balance the training split only
- keep validation and test at their natural class ratios

### Sampling rules

- image runs are spatial by construction
- video `single` mode samples one frame from a raw video
- video `sequence` mode samples contiguous ordered clips
- train clip sampling is random contiguous
- evaluation clip sampling is center contiguous

## Active Image Training Surface

The active image branch now covers:

- ViT: `IMG-EXP-01..03`
- ConvNeXt: `IMG-EXP-04..06`
- Swin: `IMG-EXP-07..08`
- DeiT: `IMG-EXP-09`
- ConvNeXtV2: `IMG-EXP-10..11`
- MaxViT: `IMG-EXP-12`
- EVA: `IMG-EXP-13..14`

This is the active image family surface for the current paper phase.

## Shared Image Optimization Stack

The active image families currently share one optimization recipe on purpose.

Current shared stack:

- optimizer: `AdamW`
- base learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR`
- minimum learning rate: `1e-6`
- warmup: `1` epoch
- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- EMA enabled with decay `0.999`
- gradient clipping at `1.0`
- best-checkpoint metric: `val_f1`
- early stopping patience: `3`
- mixed precision on CUDA

Current fixed baseline budget:

- epochs: `10`
- batch size: `64`

Why this is held fixed first:

- architecture comparison is clearer under one recipe
- optimizer drift is reduced as a hidden confound
- family-specific tuning is postponed until baseline ranking is stable

## Runtime Optimizations

The image runtime also uses:

- cuDNN benchmark on CUDA
- TF32 where supported
- float32 matmul precision set to `high` where supported
- channels-last tensor layout for 4D image batches on CUDA
- pinned-memory DataLoaders
- persistent workers when workers are enabled
- configurable worker count and prefetch factor

## Run Recording

Current image run directory layout:

- `train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/`

Expected per-run artifacts include:

- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`
- `best.pth`
- `last.pth`

Important audit caveat:

- historical saved run folders are not perfectly uniform in artifact completeness
- reporting should cite the files actually present in each run directory

## Active Video Training Truth

The video side has advanced in structure and experiment organization.

What is already active:

- category-specific registry files
- category-specific runner modules
- ordered experiment IDs
- dataset-scope resolution
- save-path resolution
- real timm-backed training for image-style video backbones
- separate smoke validation path for pipeline checks

Operational video training methodology already enforced by the active dataloader:

- identity-aware split first
- train-only oversampling second
- untouched validation split
- untouched test split

Why this ordering is mandatory:

- identity leakage is a more serious methodological failure than class imbalance
- balancing before split construction would contaminate evaluation logic
- balancing validation or test would make reported generalization weaker and less realistic

What is not yet complete:

- dedicated native-video backbone support for the reserved `VID-ST-07..12` range

That distinction must remain explicit in documentation.

## What The Thesis Or Paper Should Say

The training chapter should state clearly that:

- image and video training are treated as distinct protocol families
- the active image tree was migrated into `train/image/`
- the image family surface was updated to ViT, ConvNeXt, Swin, DeiT, ConvNeXtV2, MaxViT, and EVA
- the active video research tree was reorganized into `spa`, `tmp`, and `st`
- current video runners resolve ordered experiment IDs and execute a real timm-backed trainer for image-style video backbones
- reserved native-video IDs `VID-ST-07..12` are intentionally inactive in the current runner surface
- raw-video-first methodology remains the intended main video path
