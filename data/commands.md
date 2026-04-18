# Commands

This file records the dataset and data-pipeline commands that are currently usable in this repository, along with the commands already used during cleanup and verification.

Training-side commands now live separately in:

- `train/image/image_commands.md`
- `train/video/video_commands.md`

Old dataset helper scripts that were previously under `datasets/` have been moved to `datasets/temp/`. They should be treated as old / legacy utilities, not as the primary active data pipeline.

## Dataset Audit

Check on-disk stats against dataloader discovery:

```bash
python -m data.dataset_analyzer
```

Optional custom root:

```bash
python -m data.dataset_analyzer --root datasets
```

## Dataset Validation

Validate every dataloader-discovered sample by actually reading it:

```bash
python -m data.dataset_run
```

Use multiple workers:

```bash
python -m data.dataset_run --num-workers 16
```

Validate only image files:

```bash
python -m data.dataset_run --dtype image --num-workers 16
```

Validate only video files:

```bash
python -m data.dataset_run --dtype video --num-workers 8
```

Stop at the first failure:

```bash
python -m data.dataset_run --fail-fast --num-workers 8
```

Quick limited run:

```bash
python -m data.dataset_run --limit 100 --num-workers 8
```

Inspect true video frame counts and durations:

```bash
python -m data.video_frame_stats
```

Inspect images, videos, and extracted frame folders together:

```bash
python -m data.image_video_frame
```

Include raw readability checks:

```bash
python -m data.image_video_frame --validate
```

Smoke-test torch loading for image/video/frame media:

```bash
python -m data.image_video_frame --smoke-loader
```

Materialize derived frame folders from dataloader-discovered videos:

```bash
python -m proc.pre_process_videos
```

Quick bounded smoke run:

```bash
python -m proc.pre_process_videos --datasets celeb-df-v2 --max-frames 32 --limit 100
```

Training-oriented derived frame extraction:

```bash
python -m proc.pre_process_videos --max-frames 64
```

Important semantics:

- `--max-frames 64` means at most `64` total frames per video, sampled across the full video
- `--frame-stride 1 --max-frames None` means all frames from each video
- rerunning the same command is continuation-safe at the video-folder level because complete folders are skipped and partial/stale folders are refreshed
- this is auxiliary preprocessing, not required for the main raw-video training path

Inspect one video dataset only:

```bash
python -m data.video_frame_stats --datasets celeb-df-v2
python -m data.video_frame_stats --datasets faceforensics++
```

## Dataset Cleanup

Report unreadable files without changing anything:

```bash
python -m data.dataset_fix --action report --num-workers 16
```

Quarantine unreadable files:

```bash
python -m data.dataset_fix --action quarantine --num-workers 16
```

Delete unreadable files:

```bash
python -m data.dataset_fix --action delete --num-workers 16
```

Quarantine to a custom directory:

```bash
python -m data.dataset_fix --action quarantine --quarantine-dir datasets_quarantine --num-workers 16
```

## Commands Used In This Project Cleanup

Analyzer run before deleting corrupted images:

```bash
python -m data.dataset_analyzer
```

Full validation run that found 12 unreadable images:

```bash
python -m data.dataset_run --num-workers 16
```

Cleanup run that deleted those 12 unreadable images:

```bash
python -m data.dataset_fix --action delete --num-workers 16
```

Analyzer run after deleting corrupted images:

```bash
python -m data.dataset_analyzer
```

## Training-Related Loader Usage

These are loader modes supported by `data.dataloader.DatasetBuilder.get_loaders(...)`.

Image-only protocol:

- `dtype="image"`
- `protocol="image_only"`

Video-only protocol:

- `dtype="video"`
- `protocol="video_only"`
- default split intent: `70/10/20` with identity-aware grouping
- `mode="single"` = spatial baseline on raw videos
- `mode="sequence"` = spatial+temporal raw-video path

Optional combined auxiliary protocol:

- `protocol="combined_aux"`

Frame-folder protocol:

- `dtype="frame"`
- `protocol="frame_only"`
- default split intent: `70/10/20` with identity-aware grouping

Dataset-specific selection is supported through:

- `dataset_names=["cifake"]`
- `dataset_names=["ai-generated-images-vs-real-images"]`
- `dataset_names=["celeb-df-v2"]`
- `dataset_names=["faceforensics++"]`

## Example Python Snippets

Train on CIFAKE only:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=32,
    mode="single",
    dtype="image",
    protocol="image_only",
    dataset_names=["cifake"],
)
```

Train on AI-generated-images-vs-real-images only:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=32,
    mode="single",
    dtype="image",
    protocol="image_only",
    dataset_names=["ai-generated-images-vs-real-images"],
)
```

Train on Celeb-DF only:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=8,
    mode="sequence",
    seq_len=8,
    dtype="video",
    protocol="video_only",
    dataset_names=["celeb-df-v2"],
)
```

Train on Celeb-DF only as a spatial baseline:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=8,
    mode="single",
    dtype="video",
    protocol="video_only",
    dataset_names=["celeb-df-v2"],
)
```

Train on Celeb-DF only with explicit random-train / center-eval clip policy:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=8,
    mode="sequence",
    seq_len=8,
    clip_sampling_train="random",
    clip_sampling_eval="center",
    dtype="video",
    protocol="video_only",
    dataset_names=["celeb-df-v2"],
)
```

Train on FaceForensics++ only:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=8,
    mode="sequence",
    seq_len=8,
    dtype="video",
    protocol="video_only",
    dataset_names=["faceforensics++"],
)
```

Train on extracted frame folders only:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=8,
    mode="sequence",
    seq_len=8,
    dtype="frame",
    protocol="frame_only",
    dataset_names=["celeb-df-v2"],
)
```

Auxiliary combined image experiment:

```python
from data.dataloader import DatasetBuilder

builder = DatasetBuilder(root="datasets")
train_loader, val_loader, test_loader = builder.get_loaders(
    batch_size=32,
    mode="single",
    dtype="image",
    protocol="combined_aux",
    dataset_names=["cifake", "ai-generated-images-vs-real-images"],
)
```

## Notes

- `image_only` preserves source image train/test boundaries and derives validation from training data.
- `video_only` performs identity-aware `70/10/20` splitting for video datasets by default.
- `video_only` with `mode="single"` is the spatial baseline on raw video data.
- `video_only` with `mode="sequence"` is the main spatial+temporal video path.
- `frame_only` performs identity-aware `70/10/20` splitting for extracted frame-folder datasets by default.
- `combined_aux` is optional and should not be treated as the main paper protocol.
- separate image and video models are the intended primary research path; multimodel or mixed-media runs are auxiliary.
- raw-video training does not require offline frame extraction.
- Before every training run, it is recommended to run:

```bash
python -m data.dataset_analyzer
python -m data.dataset_run --dtype image --num-workers 16
python -m data.dataset_run --dtype video --num-workers 8
```

## Pipeline Smoke Checks

Image pipeline smoke test:

```bash
python -m train.image.simulate_image_train
```

Single-dataset smoke tests:

```bash
python -m train.image.simulate_image_train --datasets cifake
python -m train.image.simulate_image_train --datasets ai-generated-images-vs-real-images
```
