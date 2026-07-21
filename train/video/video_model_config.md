# Video Model Config

This file documents the active video experiment configuration and registry.
The code sources of truth are:

- `train/video/video_models.py`
- `train/video/video_train_backbone.py`
- `train/video/video_commands.md`

## Global Policy

- protocol: `video_only`
- raw video source; no pre-materialized image dataset is required
- default epochs: `10`
- balanced train sampling: enabled for `loss_mode=none`
- split strategy: identity-aware `70/10/20` per dataset
- train clips: random contiguous sampling
- validation/test clips: center contiguous sampling
- default decode backend: `cv2`
- default workers: `8`
- default prefetch factor: `4`

## Dataset Scopes

- `celebdf` -> `celeb-df-v2`
- `ffpp` -> `faceforensics++`
- `video_combined` -> `celeb-df-v2` + `faceforensics++`
- `real_ai_videos` -> `real-ai-videos`
- `video_all` -> all three video datasets

## Save Layout

```text
train/video/<category>/<family>/<experiment_run>/
```

Each completed run should retain `config.json`, `final_summary.json`,
`best.pth`, `last.pth`, `history.csv`, split metadata, predictions, evaluation,
and `train.log`.

## Shared Optimization Stack

- optimizer: `AdamW`
- scheduler: cosine annealing with warmup
- default base learning rate: `1e-4`
- weight decay: `1e-4`
- minimum learning rate: `1e-6`
- warmup epochs: `1`
- gradient clipping: `1.0`
- label smoothing: `0.1` for standard cross-entropy modes
- early stopping patience: `3`
- best checkpoint metric: validation F1
- mixed precision and CUDA runtime tuning enabled when available

Loss modes:

- `none`: train-only oversampling with standard cross-entropy
- `weighted_ce`: natural split with class-weighted cross-entropy
- `focal`: class-weighted focal loss
- `weighted_bce` and `focal_loss`: compatibility aliases

## Active Video Families

### Spatial

Single sampled frame per video; spatial forensic evidence only.

- `VID-SPA-01..03`: Xception/ConvNeXt scaling
- `VID-SPA-04..05`: ConvNeXtV2 scaling
- `VID-SPA-06..07`: Swin scaling
- `VID-SPA-08`: ViT
- `VID-SPA-09`: EVA
- `VID-SPA-10..11`: MaxViT scaling

### Temporal

Ordered contiguous clips with temporal aggregation or temporal heads.

- `VID-TMP-01..02`: ConvNeXt sequence baselines
- `VID-TMP-03`: ConvNeXtV2 sequence
- `VID-TMP-04..05`: Swin sequence
- `VID-TMP-06`: MaxViT sequence
- `VID-TMP-07`: ConvNeXt-LSTM
- `VID-TMP-08`: ConvNeXt temporal Transformer
- `VID-TMP-09`: ConvNeXt TCN

### Spatiotemporal

Clip-based spatial feature extraction with temporal fusion.

- `VID-ST-01`: Xception Hybrid
- `VID-ST-02..03`: ConvNeXt Hybrid
- `VID-ST-04`: Swin Hybrid
- `VID-ST-05..06`: MaxViT Hybrid scaling
- `VID-ST-07`: ConvNeXt ConvLSTM
- `VID-ST-08`: ConvNeXt Hybrid Transformer
- `VID-ST-09`: ConvNeXt Hybrid TCN

## Registry Field Values

Allowed category values:

- `spatial`
- `temporal`
- `spatiotemporal`

Allowed model modes:

- `single`: one sampled frame per video
- `sequence`: an ordered clip of frames

Allowed temporal heads implemented by the trainer:

- `mean`
- `lstm`
- `gru`
- `transformer`
- `tcn`
- `convlstm`

The registry supplies the default `batch_size` and `seq_len` for every
experiment. CLI overrides are documented in `video_commands.md` and are saved
in each run's `config.json`.
## Runtime Notes

- `run_video.py` is the canonical experiment-driven entry point.
- Category runner modules remain compatibility aliases.
- `evaluate_video.py` reconstructs saved architectures from `config.json`.
- `validate_video_dataset.py` audits raw-video decoding through the same dataset path.
- Reserved native-video IDs `VID-ST-10..12` are not active.

## Interpretation Caveat

The current `VID-TMP-02` and `VID-ST-03` runs use the same active clip-based
trainer path with per-frame backbone encoding and temporal mean pooling. The
spatiotemporal label describes the configured clip aggregation branch; it is
not a native 3D video transformer.

## Ordered Experiment Registry

This file is the thesis-facing source-of-truth ordered list for the active video experiment registry.

Ordering rule:

- category first
- then paradigm
- then family
- then parameter scale

Current registry source of truth:

- `train/video/video_models.py`

## Active Registry Summary

The current active runner registry includes:

- spatial: `VID-SPA-01..11`
- temporal: `VID-TMP-01..09`
- spatiotemporal: `VID-ST-01..09`

Reserved but inactive IDs:

- `VID-ST-10..12`

## Spatial

| experiment_id | family | model_name | params | role | category |
| --- | --- | --- | --- | --- | --- |
| `VID-SPA-01` | Xception | `xception71` | `~84M` | Baseline | spatial |
| `VID-SPA-02` | ConvNeXt | `convnext_base` | `~88M` | Baseline | spatial |
| `VID-SPA-03` | ConvNeXt | `convnext_large` | `~198M` | Scaling | spatial |
| `VID-SPA-04` | ConvNeXtV2 | `convnextv2_base.fcmae_ft_in22k_in1k` | `~89M` | Baseline | spatial |
| `VID-SPA-05` | ConvNeXtV2 | `convnextv2_large.fcmae_ft_in22k_in1k` | `~198M` | Scaling | spatial |
| `VID-SPA-06` | Swin | `swin_base_patch4_window7_224` | `~88M` | Probe | spatial |
| `VID-SPA-07` | Swin | `swin_large_patch4_window7_224` | `~197M` | Scaling | spatial |
| `VID-SPA-08` | ViT | `vit_base_patch16_224` | `~86M` | Probe | spatial |
| `VID-SPA-09` | EVA | `eva02_base_patch14_224.mim_in22k_ft_in22k_in1k` | `~86-100M` | Probe | spatial |
| `VID-SPA-10` | MaxViT | `maxvit_base_tf_224.in1k` | `~119M` | Advanced | spatial |
| `VID-SPA-11` | MaxViT | `maxvit_large_tf_224.in1k` | `~212M` | Scaling | spatial |

## Temporal

| experiment_id | family | model_name | params | role | category |
| --- | --- | --- | --- | --- | --- |
| `VID-TMP-01` | ConvNeXt Sequence | `convnext_base` | `~88M` | Baseline | temporal |
| `VID-TMP-02` | ConvNeXt Sequence | `convnext_large` | `~198M` | Scaling | temporal |
| `VID-TMP-03` | ConvNeXtV2 Sequence | `convnextv2_base.fcmae_ft_in22k_in1k` | `~89M` | Baseline | temporal |
| `VID-TMP-04` | Swin Sequence | `swin_base_patch4_window7_224` | `~88M` | Probe | temporal |
| `VID-TMP-05` | Swin Sequence | `swin_large_patch4_window7_224` | `~197M` | Scaling | temporal |
| `VID-TMP-06` | MaxViT Sequence | `maxvit_base_tf_224.in1k` | `~119M` | Advanced | temporal |
| `VID-TMP-07` | ConvNeXt LSTM | `convnext_large` | `~198M + LSTM head` | Temporal baseline | temporal |
| `VID-TMP-08` | ConvNeXt Temporal Transformer | `convnext_large` | `~198M + transformer encoder head` | Temporal attention probe | temporal |
| `VID-TMP-09` | ConvNeXt TCN | `convnext_large` | `~198M + temporal convolution head` | Efficient temporal probe | temporal |

## Spatiotemporal

| experiment_id | family | model_name | params | role | category |
| --- | --- | --- | --- | --- | --- |
| `VID-ST-01` | Xception Hybrid | `xception71` | `~84M` | Baseline | spatiotemporal |
| `VID-ST-02` | ConvNeXt Hybrid | `convnext_base` | `~88M` | Baseline | spatiotemporal |
| `VID-ST-03` | ConvNeXt Hybrid | `convnext_large` | `~198M` | Scaling | spatiotemporal |
| `VID-ST-04` | Swin Hybrid | `swin_base_patch4_window7_224` | `~88M` | Probe | spatiotemporal |
| `VID-ST-05` | MaxViT Hybrid | `maxvit_base_tf_224.in1k` | `~119M` | Advanced | spatiotemporal |
| `VID-ST-06` | MaxViT Hybrid | `maxvit_large_tf_224.in1k` | `~212M` | Scaling | spatiotemporal |
| `VID-ST-07` | ConvNeXt ConvLSTM | `convnext_large` | `~198M + ConvLSTM map head` | Spatiotemporal recurrent probe | spatiotemporal |
| `VID-ST-08` | ConvNeXt Hybrid Transformer | `convnext_large` | `~198M + temporal transformer head` | Hybrid attention probe | spatiotemporal |
| `VID-ST-09` | ConvNeXt Hybrid TCN | `convnext_large` | `~198M + temporal convolution head` | Hybrid temporal-conv probe | spatiotemporal |

## Reserved Vacancies

These IDs are intentionally not active in the current registry. They remain reserved so numbering stays stable.

- `VID-ST-10`
- `VID-ST-11`
- `VID-ST-12`

## Current Execution Note

These experiment IDs and families are active in the registry and runner surface. The current category runners now execute a real timm-backed trainer for image-style video backbones on the raw-video loader path.

Native video backbone IDs `VID-ST-10..12` remain reserved for future dedicated native-video support.

Current smoke decode note:

- the smoke path supports the default loader decode path as well as optional FFmpeg decode backends
- Intel Quick Sync hardware decode can be requested with `--decode-backend ffmpeg_qsv`
- the smoke path also supports:
  - `--transform-profile smoke_fast`
  - `--workers`
  - `--prefetch-factor`
  - `--ffmpeg-output-size`
- this is a throughput feature, not a change in experiment identity

## Current Saved Result Snapshot

Repo-wide final-result scan status:

- completed `final_summary.json` files in active `train/`: `15`
- completed image runs: `5`
- completed video runs: `10`

Best saved image-domain result in the repo:

- `IMG-EXP-04` ConvNeXt-Base
  - test F1: `0.9863`
  - test accuracy: `0.9863`

Current saved video-domain leaders:

- spatial: `VID-SPA-02` ConvNeXt-Base
  - test F1: `0.7023`
  - test accuracy: `0.8566`
- temporal: `VID-TMP-02` ConvNeXt-Large sequence
  - test F1: `0.7841`
  - test accuracy: `0.9089`
- spatiotemporal: `VID-ST-03` ConvNeXt-Large hybrid
  - test F1: `0.7841`
  - test accuracy: `0.9089`

Important implementation note:

- `VID-TMP-02` and `VID-ST-03` currently use the same active clip-based trainer logic
- both categories run per-frame image-backbone encoding followed by temporal mean pooling
- this means the current `ST` branch should be interpreted as hybrid clip aggregation, not as a distinct native-video modeling class
- `VID-TMP-07..09` add explicit LSTM, Transformer, and TCN temporal heads over ConvNeXt-Large frame features
- `VID-ST-07` adds a ConvLSTM head over ConvNeXt-Large feature maps; this is the most direct new spatiotemporal probe in the active timm-backed trainer

Incomplete artifact note:

- directories without `final_summary.json` are not part of the completed evidence inventory
- they may still contain useful partial artifacts such as `history.csv` or `run_record.md`
