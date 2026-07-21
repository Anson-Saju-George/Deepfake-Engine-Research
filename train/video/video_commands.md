# Video Commands

This is the command reference for the active video pipeline. Configuration and
experiment definitions are documented in `train/video/video_model_config.md`.

## Active Modules

- training: `train.video.run_video`
- evaluation: `train.video.evaluate_video`
- dataset validation: `train.video.validate_video_dataset`
- compatibility runners:
  - `train.video.spa.run_video_spatial`
  - `train.video.tmp.run_video_temporal`
  - `train.video.st.run_video_spatiotemporal`

## Training CLI

Canonical command:

```bash
python -m train.video.run_video --exp VID-TMP-09 --dataset-scope video_all --seq-len 4 --batch-size 2 --loss-mode none --lr 5e-5
```

All arguments:

| Argument | Type | Allowed values/default | Meaning |
| --- | --- | --- | --- |
| `--exp` | string | `VID-SPA-01..11`, `VID-TMP-01..09`, `VID-ST-01..09` | Experiment registry ID. |
| `--category` | choice | `spatial`, `temporal`, `spatiotemporal` | Optional registry filter. |
| `--dataset-scope` | choice | `celebdf`, `ffpp`, `video_combined`, `real_ai_videos`, `video_all` | Dataset selection. |
| `--batch-size` | integer | Registry default | Batch-size override. |
| `--seq-len` | integer | Registry default | Contiguous clip length for video modes. |
| `--epochs` | integer | Registry default `10` | Epoch-count override. |
| `--lr` | float | Default `1e-4` | Base learning-rate override. |
| `--loss-mode` | choice | `none`, `weighted_ce`, `weighted_bce`, `focal`, `focal_loss` | Loss/imbalance strategy. |

If `--exp` or `--dataset-scope` is omitted, the runner opens an interactive
selection menu. `weighted_bce` and `focal_loss` are accepted compatibility
aliases and are normalized by the active trainer.

## Dataset Scopes

- `celebdf` -> `celeb-df-v2`
- `ffpp` -> `faceforensics++`
- `video_combined` -> `celeb-df-v2` + `faceforensics++`
- `real_ai_videos` -> `real-ai-videos`
- `video_all` -> all three video datasets

## Category Examples

```bash
python -m train.video.run_video --exp VID-SPA-02 --dataset-scope video_combined --batch-size 8 --loss-mode none --lr 5e-5
python -m train.video.run_video --exp VID-TMP-02 --dataset-scope video_combined --seq-len 4 --batch-size 2 --loss-mode none --lr 5e-5
python -m train.video.run_video --exp VID-TMP-09 --dataset-scope video_all --seq-len 4 --batch-size 2 --loss-mode none --lr 5e-5
python -m train.video.run_video --exp VID-ST-09 --dataset-scope video_all --seq-len 4 --batch-size 2 --loss-mode none --lr 5e-5
```

## Compatibility Commands

These aliases preserve older command paths. New experiments should use
`train.video.run_video`.

```bash
python -m train.video.spa.run_video_spatial --exp VID-SPA-02 --dataset-scope video_combined
python -m train.video.tmp.run_video_temporal --exp VID-TMP-02 --dataset-scope video_combined --seq-len 4
python -m train.video.st.run_video_spatiotemporal --exp VID-ST-03 --dataset-scope video_combined --seq-len 4
```

## Evaluation CLI

Evaluation reconstructs the saved architecture and temporal head from each
run's `config.json`. It does not retrain or modify checkpoints.

```bash
python -m train.video.evaluate_video --list-runs
python -m train.video.evaluate_video --workers 0 --batch-size 2
python -m train.video.evaluate_video --run-dir train/video/tmp/ConvNeXt_TCN/<run>
```

| Argument | Type | Allowed values/default | Meaning |
| --- | --- | --- | --- |
| `--run-dir` | repeatable path | Any completed run directory | Evaluate exact run path. |
| `--checkpoint` | choice | `auto`, `best`, `last` | Checkpoint selection. |
| `--batch-size` | integer | `None` | Evaluation batch override. |
| `--workers` | integer | Runtime default | DataLoader worker count. |
| `--prefetch-factor` | integer | Runtime default | DataLoader prefetch setting. |
| `--overwrite` | flag | Off by default | Recreate prediction/evaluation artifacts. |
| `--list-runs` | flag | Off by default | List completed video runs and exit. |

Evaluation writes `test_predictions.csv` and `test_evaluation.json` beside each
completed run.

## Dataset Validation CLI

```bash
python -m train.video.validate_video_dataset --dataset-scope video_all --mode both --decode-backend cv2
```

| Argument | Type | Allowed values/default | Meaning |
| --- | --- | --- | --- |
| `--dataset-scope` | choice | Same five scopes above | Named dataset scope. |
| `--datasets` | list | Explicit dataset names | Overrides `--dataset-scope`. |
| `--mode` | choice | `single`, `sequence`, `both` | Decode paths to audit. |
| `--seq-len` | integer | Default `8` | Sequence audit clip length. |
| `--decode-backend` | choice | `auto`, `decord`, `decord_cpu`, `cv2`, `ffmpeg`, `ffmpeg_qsv`, `ffmpeg_d3d11va` | Decoder to test. |
| `--limit` | integer | `None` | Optional raw-video limit. |
| `--log-path` | path | `train/video/bad_video_log.jsonl` | JSONL failure log destination. |

## Runtime Outputs

Each completed run should contain:

- `config.json`
- `train.log`
- `best.pth` and `last.pth`
- `best_summary.json` and `final_summary.json`
- `history.csv`
- `split_summary.json`
- `test_predictions.csv`
- `test_evaluation.json`
- `run_record.md`
