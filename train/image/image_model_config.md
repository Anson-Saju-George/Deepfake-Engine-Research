# Image Model Config

This file documents the active image experiment ladder implemented in `train/image/`.

The code source of truth is:

- `train/image/image_models.py`
- `train/image/image_commands.md`

## Global Policy

- protocol: `image_only`
- default epochs: `10`
- default batch size: `64`
- balanced sampling: enabled
- primary datasets:
  - `cifake`
  - `ai-generated-images-vs-real-images`
  - combined image setting using both
- image split rule:
  - preserve source dataset train and test boundaries
  - derive validation from the source training partition
  - do not force a pooled synthetic `70/10/20` split for image datasets

## Save Layout

```text
train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/
```

Dataset tags:

- `cifake`
- `ai_gen`
- `image_combined`

## Shared Optimization Stack

All active image families use the same core optimization defaults:

- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- base learning rate: `1e-4`
- weight decay: `1e-4`
- minimum learning rate: `1e-6`
- warmup epochs: `1`
- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- EMA enabled with decay `0.999`
- gradient clipping: `1.0`
- best checkpoint metric: `val_f1`
- early stopping patience: `3`
- mixed precision on CUDA

## Active Image Families

### ViT

- `IMG-EXP-01` -> `vit_base_patch16_224` -> `~86M`
- `IMG-EXP-02` -> `vit_large_patch16_224` -> `~307M`
- `IMG-EXP-03` -> `vit_huge_patch14_224` -> `~632M`

### ConvNeXt

- `IMG-EXP-04` -> `convnext_base` -> `~89M`
- `IMG-EXP-05` -> `convnext_large` -> `~198M`
- `IMG-EXP-06` -> `convnext_xlarge` -> `~350M`

### Swin

- `IMG-EXP-07` -> `swin_base_patch4_window7_224` -> `~88M`
- `IMG-EXP-08` -> `swin_large_patch4_window7_224` -> `~197M`

### DeiT

- `IMG-EXP-09` -> `deit3_base_patch16_224` -> `~86M`

### ConvNeXtV2

- `IMG-EXP-10` -> `convnextv2_base.fcmae_ft_in22k_in1k` -> `~89M`
- `IMG-EXP-11` -> `convnextv2_large.fcmae_ft_in22k_in1k` -> `~198M`

### MaxViT

- `IMG-EXP-12` -> `maxvit_base_tf_224.in1k` -> `~119M`

### EVA

- `IMG-EXP-13` -> `eva02_base_patch14_224.mim_in22k_ft_in22k_in1k` -> `~86M`
- `IMG-EXP-14` -> `eva02_large_patch14_224.mim_m38m_ft_in22k_in1k` -> `~304M`

## Runtime Notes

- tqdm is kept visible separately for train, validation, and test phases.
- mixed precision is used only on CUDA paths.
- channels-last, pinned memory, and worker prefetching are part of the current image runtime tuning.
- each runner now writes a full CLI transcript to `train.log` inside the run directory.

## Current Saved Result Snapshot

Completed saved image runs currently visible in the repo:

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

Repo-wide audit note:

- these `5` runs are the full set of repo-visible completed image `final_summary.json` files under the active `train/` tree

Current image conclusions:

- `IMG-EXP-04` ConvNeXt-Base is the strongest completed saved image run
- `IMG-EXP-07` Swin-Base is competitive but not ahead
- larger did not automatically improve results, because `IMG-EXP-05` did not beat `IMG-EXP-04`
- the completed ViT runs lag behind the strongest CNN and hierarchical-transformer image runs

Research interpretation:

- the current image benchmark is close to saturation relative to the present datasets
- this means image-domain results should be presented as strong artifact-detection benchmarks, not as proof that raw-video detection is equally solved

## Artifact Expectations

Each completed image run is expected to keep:

- `config.json`
- `history.csv`
- `best.pth`
- `last.pth`
- `best_summary.json`
- `split_summary.json`
- `run_record.md`

Additional promoted final artifacts may exist when the run reaches the configured promotion threshold.

Documentation caveat:

- `best_summary.json` is not perfectly uniform across historical runs
- `final_summary.json` and `run_record.md` are the safer result anchors for thesis reporting

## Naming Notes

- `train/image/image_models.py` is the active registry filename.
- `train/image/image_models.py` is retained only as a compatibility shim.
