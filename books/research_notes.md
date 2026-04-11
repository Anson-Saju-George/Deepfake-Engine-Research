# Research Notes

## Scope

This file captures high-level project decisions, rationale, caveats, and experimental notes that are useful for research reporting but should not overload the operational documentation.

The active implementation anchors are:

- `data/dataloader.py`
- `data/dataset.md`
- `data/commands.md`
- `proc/pre_process_videos.py`
- `train/`
- `train_old/` for legacy reference, especially video

## Current Repo Truth

- Active image training tree: `train/`
- Legacy reference training tree: `train_old/`
- Active dataloader: `data/dataloader.py`
- Active image protocol: `image_only`
- Active video protocol: `video_only`
- Auxiliary mixed-media protocol: `combined_aux`
- Derived frame-folder protocol: `frame_only`

The current design intentionally separates image and video experiments by default. Mixed image+video loading is not the default training path.

Operational interpretation:

- image-only experiments are the image-domain spatial benchmark family
- video-only `single` mode is the video-domain spatial baseline
- video-only `sequence` mode is the video-domain spatial+temporal path

Current implementation status:

- image smoke validation is active in the new tree via `python -m train.image_simulate`
- ViT image training is active in the new tree via `python -m train.run_image_vit`
- video execution in the new tree is still pending, although the planning registries exist

## Dataset Snapshot

Latest cleaned raw dataset state:

- Total raw samples: `192,016`
- Images: `179,988`
- Videos: `12,028`
- Labels: `99,740 fake`, `92,276 real`

Historical cleaning notes:

- `12` corrupted images were deleted from `ai-generated-images-vs-real-images`
- `3` bad videos had been deleted earlier due to old `moov`-type issues

Important note:

- During frame extraction, dataloader-visible totals can rise above `192,016` because extracted frame folders become discoverable as `dtype="frame"` records in addition to the raw image and video corpora.
- `data.dataset_analyzer` now reports the raw image and raw video corpora used by the main training path, so its raw total remains `192,016` even when auxiliary frame folders exist elsewhere on disk.

## Split Policy Decisions

### Images

- Preserve original source dataset `train/test` boundaries.
- Derive validation from the source `train` split when no explicit validation split exists.
- Do not globally pool and re-split image datasets into a synthetic `70/10/20`.

### Videos

- Use identity-aware grouping for split assignment.
- Default target split is `70/10/20` for `train/val/test`.
- Treat `single` as the spatial baseline and `sequence` as the spatial+temporal path.

### Frame Folders

- Treat extracted frame folders as derived video data, not as an independent native benchmark.
- Use identity-aware `70/10/20`.

## Sampling Decisions

- Image models are spatial by construction.
- Video `mode="single"` samples one frame from a raw video and acts as the spatial baseline on video data.
- Video `mode="sequence"` uses contiguous clip sampling.
- Video `mode="sequence"` retains spatial information within each frame while adding temporal information across ordered frames.
- Training clip sampling uses random contiguous clips.
- Evaluation clip sampling uses center contiguous clips.
- `frame_stride=1` in preprocessing means every frame is extracted.
- `max_frames=None` means no per-video frame cap is applied.

## Video Distribution Notes

Observed video datasets are fake-heavy by file count:

- `celeb-df-v2`
- `faceforensics++`

Frame-mass analysis helps characterize imbalance more precisely, but it does not eliminate the class skew in the major video datasets.

Key observation:

- `real-ai-videos` is tiny and balanced by video count, with real-dominant frame mass in current analysis.
- The two main video datasets remain the critical drivers of overall video imbalance.

Research implication:

- Class balancing should be handled during training after dataset boundaries and identity-aware splits are correct.

## Preprocessing Notes

Accepted active path:

- `proc/pre_process_videos.py`

Reasons:

- aligned with the active dataloader
- continuation-safe via per-folder manifests
- appropriate for generating derived frame folders for frame-only or auxiliary experiments
- not required for the main raw-video training path

Rejected or deferred path:

- dedicated NVIDIA GPU extraction path was evaluated and not adopted into the active repo
- sampled tests on `celeb-df-v2` showed MPEG-4 videos and failed CUDA decode device setup in the tested FFmpeg path
- CPU extraction remains the accepted preprocessing route

## Training Structure

Intended main experiments:

- separate image-only models
- separate video-only spatial baselines
- separate video-only spatial+temporal models
- optional frame-only experiments

Auxiliary only:

- combined image+video loading
- multi-branch or pseudo-multimodal runs unless explicitly justified as an ablation

## Current New-Tree Image Record

Verified completed run:

- experiment: `IMG-EXP-01`
- model: `vit_base_patch16_224`
- dataset scope: `image_combined`
- best epoch: `6`
- best validation F1: `0.972966`
- best validation accuracy: `0.972792`
- final test accuracy: `0.970278`
- final test F1: `0.970092`

Implemented current image save layout:

- `train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/`

Current implemented family path example:

- `train/image/ViT/IMG-EXP-01_vit_base_patch16_224_image_combined/`

## Paper Caveats

- Extracted frame folders are derived from raw videos and should be reported as such.
- Identity-aware split quality depends on the naming conventions present in each source dataset.
- Full-frame extraction is computationally expensive and should be described separately from model training.
- Report balancing and augmentation decisions independently from raw corpus composition.

## Near-Term Priorities

- complete final video/frame distribution analysis
- keep frame extraction auxiliary and storage-aware
- run separate image-only and video-only baselines
- document experiment matrix and evaluation methodology before large training sweeps
