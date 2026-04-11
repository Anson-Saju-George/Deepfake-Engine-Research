# Model Selection

## Active Families

### Image Models

- ViT presets
- ConvNeXt presets
- EfficientNet presets
- ResNet presets
- current implemented new-tree execution:
  - ViT only, `IMG-EXP-01..03`

### Video Models

- baseline frame-aggregation configurations
- dataset-specific ConvNeXt-based presets already defined in `train_old/video_models/model_configs.py`

## Selection Principles

- start with strong, reproducible baselines
- keep image and video model families conceptually separate
- avoid introducing multimodal complexity before baseline validity is established

## Current Decision

The main paper path should emphasize:

- image-only benchmarks
- video-only benchmarks

Current implementation order:

1. image pipeline smoke validation
2. ViT image baselines in the new tree
3. remaining image families
4. migrated video training execution in the new tree

Frame-only and combined auxiliary experiments should follow after the baseline matrix is stable.
