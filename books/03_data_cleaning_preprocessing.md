# Data Cleaning & Preprocessing

## Cleaning History

- `12` corrupted images were removed from `ai-generated-images-vs-real-images`
- `3` bad videos were removed historically due to old `moov`-type issues

Validation evidence already documented in the dataset audit trail:

- pre-image-cleanup total: `192,028`
- post-image-cleanup total: `192,016`
- the `12` detected failures were image-only failures from `ai-generated-images-vs-real-images`
- there is no current evidence from the latest validation pass of additional unreadable videos in the checked `12,028` raw video files

## Active Preprocessing Path

- `proc/pre_process_videos.py`

This extractor:

- discovers videos via `DatasetBuilder`
- writes continuation-safe frame folders
- stores per-folder extraction manifests
- supports full-frame extraction and bounded extraction modes

Current accepted interpretation:

- raw-video training does not require frame materialization
- frame extraction is an optional derived-data step for frame-only or auxiliary experiments
- the active extractor is continuation-safe, so reruns should skip complete outputs and refresh stale or partial ones

## Important Semantics

- `frame_stride=1` means every frame
- `max_frames=None` means all eligible frames after stride
- reruns skip complete outputs and refresh stale or partial ones
- bounded extraction such as `--max-frames 32` or `--max-frames 64` remains the practical training-oriented option when a smaller derived frame dataset is desired

## Validation and Cleanup Tooling

Relevant active tooling:

- `python -m data.dataset_run`
- `python -m data.dataset_fix`
- `python -m data.dataset_analyzer`

Operational interpretation:

- use `dataset_run` to detect unreadable media
- use `dataset_fix` to report, quarantine, or delete unreadable media
- use `dataset_analyzer` after cleanup to confirm final counts and loader agreement

## Deferred / Rejected Routes

- GPU-specific extraction was evaluated but not adopted into the active repo
- CPU extraction remains the accepted baseline preprocessing route

## Legacy Utilities

Old helper scripts formerly under `datasets/` were moved to `datasets/temp/` and should be treated as legacy utilities, not the active pipeline.

Examples retained only for traceability:

- `datasets/temp/extract_ff_frames.py`
- `datasets/temp/frame_converter.py`
- `datasets/temp/clean_ds.py`
- `datasets/temp/del_corrupted.py`
- `datasets/temp/rgba_clean.py`
- `datasets/temp/remove_audio.py`
