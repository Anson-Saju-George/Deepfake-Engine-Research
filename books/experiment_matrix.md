# Experiment Matrix

## Purpose

This file maps the intended experiment surface across protocols, datasets, split logic, entrypoints, and expected artifacts.

## Active Experiment Families

| Family | Protocol | Media Type | Dataset Scope | Split Strategy | Sampling | Entrypoint | Config Key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Image Smoke Test | `image_only` | image | CIFAKE | preserve source train/test, val from train | single image, spatial-only | `python -m train.image_simulate --datasets cifake` | smoke test |
| Image Smoke Test | `image_only` | image | AI-generated-images-vs-real-images | preserve source train/test, val from train | single image, spatial-only | `python -m train.image_simulate --datasets ai-generated-images-vs-real-images` | smoke test |
| Image ViT | `image_only` | image | CIFAKE | preserve source train/test, val from train | single image, spatial-only | `python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope cifake` | `IMG-EXP-01..03` |
| Image ViT | `image_only` | image | AI-generated-images-vs-real-images | preserve source train/test, val from train | single image, spatial-only | `python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope ai_gen` | `IMG-EXP-01..03` |
| Image ViT | `image_only` | image | CIFAKE + AI-generated-images-vs-real-images | preserve per-dataset source train/test, val from train | single image, spatial-only | `python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope image_combined` | `IMG-EXP-01..03` |
| Video Spatial Baseline | `video_only` | raw video | Celeb-DF v2 | identity-aware `70/10/20` | `mode="single"` sampled frame from raw video | `train_old/video_models/run.py` | current loader-supported baseline |
| Video Spatial Baseline | `video_only` | raw video | FaceForensics++ | identity-aware `70/10/20` | `mode="single"` sampled frame from raw video | `train_old/video_models/run.py` | current loader-supported baseline |
| Video Spatial+Temporal | `video_only` | raw video | Celeb-DF v2 | identity-aware `70/10/20` | `mode="sequence"` random contiguous train clip, center contiguous eval clip | `train_old/video_models/run.py` | `video_celebdf` |
| Video Spatial+Temporal | `video_only` | raw video | FaceForensics++ | identity-aware `70/10/20` | `mode="sequence"` random contiguous train clip, center contiguous eval clip | `train_old/video_models/run.py` | `video_ffpp` |
| Video Spatial+Temporal | `video_only` | raw video | Celeb-DF v2 + FaceForensics++ | identity-aware per-dataset `70/10/20` | `mode="sequence"` random contiguous train clip, center contiguous eval clip | `train_old/video_models/run.py` | `video_combined` |
| Video Auxiliary | `video_only` | raw video | real-ai-videos | identity-aware `70/10/20` | `mode="sequence"` random contiguous train clip, center contiguous eval clip | `train_old/video_models/run.py` | `video_real_ai` |
| Frame Derived | `frame_only` | frame folder | extracted video frame folders | identity-aware `70/10/20` | single frame or contiguous saved-frame sequence | future training entrypoint | future preset |
| Mixed Auxiliary | `combined_aux` | mixed | images + videos and optionally frames | grouped auxiliary split only | mixed | auxiliary only | task-specific |

## Current Launch References

### Image

- smoke test:
  - `python -m train.image_simulate`
- ViT runner:
  - `python -m train.run_image_vit`
- current dataset-scope tags:
  - `cifake`
  - `ai_gen`
  - `image_combined`

### Video

- Celeb-DF-only: `video_celebdf`
- FaceForensics++-only: `video_ffpp`
- combined video: `video_combined`
- small auxiliary video: `video_real_ai`

## Research Reporting Notes

- Image and video results should be presented as separate benchmark families.
- Video results should distinguish spatial baselines from spatial+temporal sequence models.
- Frame-derived experiments should be labeled clearly as derived-video experiments.
- `combined_aux` should be reported as auxiliary or ablation-only, not as the default core protocol.

## Matrix Gaps

- Only the ViT branch is implemented in the new top-level image trainer so far.
- ConvNeXt, EfficientNet, and ResNet trainers in the new tree still need implementation.
- New video trainer execution code is not implemented yet; only the registry/planning layer exists.
- No dedicated `frame_only` training preset tree has been formalized yet.
