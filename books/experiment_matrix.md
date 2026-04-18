# Experiment Matrix

## Purpose

This file maps the active experiment surface across protocols, datasets, split logic, entrypoints, and implementation status.

## Active Matrix

| Family | Protocol | Media Type | Dataset Scope | Split Strategy | Sampling | Entrypoint | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Image Smoke Test | `image_only` | image | `cifake` | preserve source train/test, val from train | single image, spatial-only | `python -m train.image.simulate_image_train --datasets cifake` | active |
| Image Smoke Test | `image_only` | image | `ai-generated-images-vs-real-images` | preserve source train/test, val from train | single image, spatial-only | `python -m train.image.simulate_image_train --datasets ai-generated-images-vs-real-images` | active |
| Image ViT | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_vit` | active |
| Image ConvNeXt | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_convnext` | active |
| Image Swin | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_swin` | active |
| Image DeiT | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_deit` | active |
| Image ConvNeXtV2 | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_convnextv2` | active |
| Image MaxViT | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_maxvit` | active |
| Image EVA | `image_only` | image | `cifake`, `ai_gen`, `image_combined` | preserve per-dataset source boundaries | single image, spatial-only | `python -m train.image.run_image_eva` | active |
| Video Smoke | `video_only` | raw video | `celeb-df-v2`, `faceforensics++`, optional combined or `real-ai-videos` | identity-aware per dataset | `single`, `sequence`, or both | `python -m train.video.simulate_video_train` | active |
| Video Spatial Registry | `video_only` | raw video | `celebdf`, `ffpp`, `video_combined`, `real_ai_videos`, `video_all` | identity-aware per dataset | single sampled frame | `python -m train.video.spa.run_video_spatial` | active runner surface, real timm-backed trainer for image-style backbones |
| Video Temporal Registry | `video_only` | raw video | `celebdf`, `ffpp`, `video_combined`, `real_ai_videos`, `video_all` | identity-aware per dataset | contiguous clip | `python -m train.video.tmp.run_video_temporal` | active runner surface, real timm-backed trainer for image-style backbones |
| Video Spatiotemporal Registry | `video_only` | raw video | `celebdf`, `ffpp`, `video_combined`, `real_ai_videos`, `video_all` | identity-aware per dataset | contiguous clip aggregation | `python -m train.video.st.run_video_spatiotemporal` | active runner surface, real timm-backed trainer for image-style backbones; reserved native-video IDs remain inactive |
| Frame Derived | `frame_only` | frame folder | derived frame roots | identity-aware `70/10/20` | single frame or contiguous saved-frame sequence | task-specific future entrypoint | auxiliary |
| Mixed Auxiliary | `combined_aux` | mixed | images plus videos and optionally frames | grouped auxiliary split only | mixed | task-specific | auxiliary |

## Current Image Experiment Surface

- `IMG-EXP-01..03`: ViT
- `IMG-EXP-04..06`: ConvNeXt
- `IMG-EXP-07..08`: Swin
- `IMG-EXP-09`: DeiT
- `IMG-EXP-10..11`: ConvNeXtV2
- `IMG-EXP-12`: MaxViT
- `IMG-EXP-13..14`: EVA

## Current Video Experiment Surface

Spatial ordered surface:

- `VID-SPA-01..11`

Temporal ordered surface:

- `VID-TMP-01..06`

Spatiotemporal ordered surface:

- active: `VID-ST-01..06`
- reserved vacancies: `VID-ST-07..12`

See the detailed ordered list in:

- `train/video/Experiment_List.md`

## Research Reporting Notes

- image and video results should be presented as separate benchmark families
- video results should distinguish spatial, temporal, and spatiotemporal categories
- frame-derived experiments should be labeled clearly as derived-video experiments
- `combined_aux` should be reported only as auxiliary or ablation work
- current video registry runners are active for the image-style video backbones in the registry
- reserved native-video IDs should not be described as active runner entries
