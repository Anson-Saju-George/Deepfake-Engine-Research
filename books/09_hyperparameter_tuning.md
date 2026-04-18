# Hyperparameter Tuning

## What This Stage Means

Hyperparameter tuning adjusts the training recipe in a controlled way after the baseline methodology is already stable.

In this repository, tuning is intentionally downstream of:

- dataset truth
- protocol truth
- split correctness
- baseline training comparability

## Current Fixed Image Baseline

The active image tree currently uses a deliberately fixed baseline so model-family comparisons remain interpretable.

Current image defaults:

- epochs: `10`
- batch size: `64`
- optimizer: `AdamW`
- base learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR`
- minimum learning rate: `1e-6`
- warmup: `1` epoch
- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- EMA decay: `0.999`
- gradient clipping: `1.0`
- early stopping patience: `3`
- minimum improvement delta: `1e-4`
- best checkpoint metric: `val_f1`

## Why The Baseline Is Held Fixed First

The repository is currently comparing these active image families:

- ViT
- ConvNeXt
- Swin
- DeiT
- ConvNeXtV2
- MaxViT
- EVA

If optimizer, scheduler, batch size, patience, and training budget all change at once, architecture comparison becomes much weaker.

That is why the current policy is:

- compare families under one shared recipe first
- identify promising families and scales
- tune more deeply later

## Main Tuning Axes In This Repo

Likely later tuning axes include:

- learning rate
- weight decay
- batch size
- epoch budget
- warmup duration
- label smoothing
- EMA decay
- gradient clipping threshold
- sequence length for video experiments
- batch size for larger video backbones
- augmentation strength
- balancing strategy
- optimizer and scheduler alternatives

## Video-Side Tuning Reality

The video registry surface is now well organized, and the active image-style video backbones do support real tuning work.

Why:

- the current video runners now execute a real timm-backed trainer for the image-style video backbones in the active registry
- reserved native-video IDs `VID-ST-07..12` remain inactive because dedicated native-video support is not implemented yet
- current video findings already show that sequence length and base learning rate can materially affect outcome quality

So the present tuning truth is:

- image tuning is meaningful later, after baseline ranking stabilizes
- video tuning is meaningful now for the active image-style backbones
- native-video tuning remains a later phase until a dedicated native-video trainer exists

Current observed useful video tuning axes from saved runs:

- lowering sequence length from `8` or `12` to `4` for heavier sequence and hybrid runs
- lowering base learning rate from `1e-4` to `5e-5` for unstable sequence and hybrid backbones
- comparing `loss_mode = none`, `weighted_ce`, and `focal` without changing split methodology

## What The Completed Tuning Evidence Actually Shows

The tuning story is no longer hypothetical. Some useful conclusions already exist.

### Image-side evidence

Current completed image runs were compared under one shared recipe rather than aggressive family-specific tuning.

That taught two important things:

- architecture choice mattered more than widening the recipe immediately
- larger scale did not automatically improve results, because `IMG-EXP-05` ConvNeXt-Large did not beat `IMG-EXP-04` ConvNeXt-Base

### Video-side evidence

The strongest completed video result came after reducing optimization aggressiveness for the larger clip-based ConvNeXt run:

- `VID-TMP-02`
  - `seq_len = 4`
  - `base_lr = 5e-5`
  - test F1: `0.7841`

That is strong evidence that:

- sequence length is a real tuning axis
- base learning rate is a real stability axis for heavier clip-based backbones
- simply scaling a model without reducing optimization aggressiveness would have been weaker

### Loss-mode evidence

Current completed loss-mode comparisons do not support a claim that more complex imbalance handling is automatically better.

Observed pattern:

- `loss_mode = none`
  - keeps train-only oversampling
  - produced the strongest spatial ConvNeXt result
- `loss_mode = weighted_ce`
  - was competitive in some runs
  - but did not produce a clear decisive win over the baseline strategy
- `loss_mode = focal`
  - underperformed materially on the completed spatial ConvNeXt comparison

Practical conclusion:

- the repo should not claim that focal loss solved the imbalance problem
- the present evidence supports keeping the simplest train-only oversampling baseline as the default reported setting

### Runtime-side tuning evidence

Operational tuning also mattered on the video side.

Current stable practical defaults were not chosen arbitrarily:

- `decode_backend = cv2`
- `gpu_aug = none`
- `workers = 8`
- `prefetch_factor = 4`

Why they survived:

- `ffmpeg_qsv` produced many logged decode failures on the mixed-quality corpus
- `decord` was useful to test, but remained unsafe as a default on corrupted H.264 data
- Kornia GPU augmentation remained optional rather than default because the biggest bottleneck was still decode and input delivery, not only CPU augmentation

## What The Thesis Or Paper Should Say

The tuning chapter should state that:

- the current image baseline was intentionally held fixed for fair architecture comparison
- family-specific tuning was postponed until after baseline ranking
- the active image family surface changed as the project matured
- the video registry is now ordered and organized
- active video tuning is meaningful for the image-style video backbones already supported by the current trainer
- reserved native-video IDs remain a later phase because dedicated native-video support is not yet implemented
