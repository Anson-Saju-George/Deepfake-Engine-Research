# Dataset Record

This file is the dataset audit trail for the repository. It records the current dataset inventory, folder layout, loader behavior, validation tooling, and the methodological caveats that must be acknowledged in experiments and in the final paper.

## Purpose

This document should answer four questions at any point in the project:

1. What datasets are physically present on disk?
2. What does the dataloader actually discover and use?
3. What scripts generated or cleaned any derived data?
4. What methodological risks exist in the current data pipeline?

## Dataset Roots

Primary dataset roots:

- `datasets/images`
- `datasets/videos`

Optional or derived roots created by preprocessing utilities:

- `datasets/preprocessed_frames`
- `datasets/images/faceforensics_frames`
- legacy outputs from older scripts: `video-frames`, `frames`

Baseline raw-only status on a clean cleaned checkout:

- `datasets/preprocessed_frames`: optional, may be absent on raw-only baselines and may appear during auxiliary experiments
- `datasets/images/faceforensics_frames`: optional legacy-derived root
- `video-frames`: optional legacy-derived root
- `frames`: optional legacy-derived root

## Current On-Disk Tree Summary

Verified top-level datasets currently present:

- `datasets/images/ai-generated-images-vs-real-images`
- `datasets/images/cifake`
- `datasets/videos/celeb-df-v2`
- `datasets/videos/faceforensics++`
- `datasets/videos/real-ai-videos`

Optional frame-derived roots that may or may not exist depending on preprocessing state:

- `datasets/preprocessed_frames`
- `datasets/images/faceforensics_frames`
- `video-frames`
- `frames`

Verified nested video tree structure:

- `celeb-df-v2`
  - `Celeb-fake`
  - `Celeb-real`
  - `YouTube-real`
- `faceforensics++`
  - `fake/DeepFakeDetection/c23/videos`
  - `fake/Deepfakes/c23/videos`
  - `real/actors/c23/videos`
  - `real/youtube/c23/videos`
- `real-ai-videos`
  - `fake`
  - `real`

## Current Progress

Latest repo progress relevant to the data pipeline:

- raw image and raw video loading are active and verified against the current cleaned dataset state
- raw-video training is the primary video path; offline frame materialization is auxiliary only
- `video_only` with `mode="single"` is the spatial baseline on raw videos
- `video_only` with `mode="sequence"` is the spatial+temporal path on raw videos
- extracted frame folders are now supported as a first-class media type through `dtype="frame"` and `protocol="frame_only"`
- `data.image_video_frame` was added to analyze images, videos, and derived frame folders together
- `proc/pre_process_videos.py` now discovers videos through `DatasetBuilder` instead of maintaining a separate scan path
- `proc/pre_process_videos.py` is continuation-safe at the video-folder level:
- complete frame folders with matching manifests are skipped on rerun
- partial or stale frame folders are refreshed automatically on rerun
- old legacy Python utilities that were sitting directly under `datasets/` were moved to `datasets/temp/` to keep the active dataset tree cleaner

Current status of derived frame extraction:

- the loader and analyzers support `datasets/preprocessed_frames` when present
- the root is optional and should not be treated as required for the default raw-video workflow
- full-corpus all-frame extraction is a materialization job and can take many hours
- full-corpus all-frame extraction can also explode storage usage and is therefore not the recommended default path
- bounded extraction such as `--max-frames 32` or `--max-frames 64` remains the practical training-oriented option when a smaller derived frame dataset is desired

## Pipeline Roles

The repository currently has three distinct data-layer roles. These should not be conflated in the paper.

### Active runtime path

- `data/dataloader.py`
- active image training tree under `train/image/`
- active video research tree under `train/video/`
- legacy references under `train_old/`
- raw image and raw video loading are the active default training path
- image datasets feed image-only spatial models
- raw video datasets feed video-only models
- `video_only` + `single` is the spatial baseline on videos
- `video_only` + `sequence` is the clip-based spatial+temporal path on videos
- optional derived frame-folder loading is supported through `dtype="frame"`, but it is auxiliary rather than part of the main counted baseline
- separate image and video models are the intended primary research path
- mixed-media loading should still be treated as auxiliary rather than as the main reported protocol
- current implementation reality:
  - `train/image/simulate_image_train.py` is the active image smoke-test entrypoint
  - active image runner families now exist for:
    - `train/image/run_image_vit.py`
    - `train/image/run_image_convnext.py`
    - `train/image/run_image_swin.py`
    - `train/image/run_image_deit.py`
    - `train/image/run_image_convnextv2.py`
    - `train/image/run_image_maxvit.py`
    - `train/image/run_image_eva.py`
  - active video registry and runner surfaces now exist under:
    - `train/video/video_models.py`
    - `train/video/spa/`
    - `train/video/tmp/`
    - `train/video/st/`
  - current video execution still routes through smoke validation rather than the final migrated research trainer

### Audit and validation path

- `data/dataset_analyzer.py`
- `data/dataset_run.py`
- `data/dataset_fix.py`
- `data/video_frame_stats.py`
- `data/image_video_frame.py`
- these scripts are used to audit filesystem truth, readability, media statistics, and loader behavior before training

### Optional materialization path

- `proc/pre_process_videos.py`
- this is an optional active utility that materializes derived frame folders from dataloader-discovered videos
- it is not required for the default raw-video training path
- it should be described as preprocessing for derived frame experiments, not as a mandatory step for the core dataset pipeline

### Legacy helper path

- everything under `datasets/temp/`
- these scripts are retained for historical traceability and reference
- they are not the primary active pipeline and should not be presented as the current canonical workflow

## Verified Counts From `python -m data.dataset_analyzer`

Historical cleanup notes:

- `3` unreadable videos had already been deleted manually in the past due to `moov`/video metadata errors
- `12` unreadable images were deleted later with `python -m data.dataset_fix --action delete --num-workers 16`

Analyzer status before deleting the 12 corrupted images:

- Dataloader scan total: `192,028`
- On-disk totals and loader totals matched for every configured dataset

Latest analyzer status after deleting the 12 corrupted images:

- Total raw discoverable samples: `192,016`
- Total images: `179,988`
- Total videos: `12,028`
- Fake labels: `99,740`
- Real labels: `92,276`
- On-disk totals and loader totals matched for every configured dataset

Important interpretation of the latest analyzer output:

- `data.dataset_analyzer` now reports the raw image and raw video corpora used by the main training path
- derived frame folders are intentionally excluded from the main raw-data totals in that script
- when derived frame folders exist on disk, mixed-media tools such as `data.image_video_frame` or ad hoc builder scans can still show a larger total because those folders appear as additional `dtype="frame"` records

### Image datasets

#### `ai-generated-images-vs-real-images`

- Type: image classification
- Structure: `train/{fake,real}` and `test/{fake,real}`
- Historical pre-cleanup counts:
- `train/fake`: `24,000`
- `train/real`: `24,000`
- `test/fake`: `6,000`
- `test/real`: `6,000`
- Historical total: `60,000`
- Deleted corrupted images:
- `train/fake`: `2`
- `train/real`: `7`
- `test/real`: `3`
- Latest on-disk counts after deletion:
- `train/fake`: `23,998`
- `train/real`: `23,993`
- `test/fake`: `6,000`
- `test/real`: `5,997`
- Latest total: `59,988`
- Loader labels:
- `fake -> 0`
- `real -> 1`
- Latest loader-discovered totals after deletion:
- fake: `29,998`
- real: `29,990`

#### `cifake`

- Type: image classification
- Structure: `train/{FAKE,REAL}` and `test/{FAKE,REAL}`
- Verified counts:
- `train/FAKE`: `50,000`
- `train/REAL`: `50,000`
- `test/FAKE`: `10,000`
- `test/REAL`: `10,000`
- Total: `120,000`
- Loader labels:
- `FAKE -> 0`
- `REAL -> 1`
- Loader-discovered totals:
- fake: `60,000`
- real: `60,000`

#### `faceforensics_frames`

- Type: derived image dataset
- Source: created by `datasets/extract_ff_frames.py`
- Expected structure: `train|val|test/{fake,real}`
- Current status: not present on disk
- Current effect on loader: ignored safely because the folder does not exist

### Video datasets

#### `celeb-df-v2`

- Type: video classification
- Structure: flat class folders
- Verified counts:
- `Celeb-fake`: `5,643`
- `Celeb-real`: `590`
- `YouTube-real`: `300`
- Total: `6,533`
- Loader mapping:
- `Celeb-fake -> 0`
- `Celeb-real -> 1`
- `YouTube-real -> 1`
- Loader-discovered totals:
- fake: `5,643`
- real: `890`

#### `faceforensics++`

- Type: video classification
- Structure used by loader: `fake/**` and `real/**`
- Verified counts:
- fake total: `4,066`
- real total: `1,363`
- nested fake subsets:
- `fake/DeepFakeDetection/c23/videos`: `3,066`
- `fake/Deepfakes/c23/videos`: `1,000`
- nested real subsets:
- `real/actors/c23/videos`: `363`
- `real/youtube/c23/videos`: `1,000`
- Total: `5,429`
- Loader mapping:
- anything under `fake -> 0`
- anything under `real -> 1`
- Loader-discovered totals:
- fake: `4,066`
- real: `1,363`

#### `real-ai-videos`

- Type: video classification
- Structure: flat class folders
- Verified counts:
- `fake`: `33`
- `real`: `33`
- Total: `66`
- Loader mapping:
- `fake -> 0`
- `real -> 1`
- `ai -> 0` if that folder appears later
- Loader-discovered totals:
- fake: `33`
- real: `33`

## Global Totals

Historical totals before deleting the 12 corrupted images:

- Total image samples on disk: `180,000`
- Total video samples on disk: `12,028`
- Total dataloader-discovered samples: `192,028`
- Fake labels: `99,742`
- Real labels: `92,286`

Current totals after deleting the 12 corrupted images:

- Total image samples on disk: `179,988`
- Total video samples on disk: `12,028`
- Total dataloader-discovered samples: `192,016`
- Fake labels: `99,740`
- Real labels: `92,276`

Historical raw video total before the earlier manual deletion of 3 bad videos:

- Raw video total before that cleanup: `12,031`
- Current video total after that cleanup: `12,028`

## Latest Analyzer Snapshot

This section reflects the most recent analyzer run after image cleanup.

Command:

- `python -m data.dataset_analyzer`

Latest discovered totals:

- Total samples discovered: `192,016`
- By dtype:
- image: `179,988`
- video: `12,028`
- By label:
- fake: `99,740`
- real: `92,276`

Latest per-dataset discovered totals:

- `ai-generated-images-vs-real-images`: `59,988`
- fake: `29,998`
- real: `29,990`
- `cifake`: `120,000`
- fake: `60,000`
- real: `60,000`
- `celeb-df-v2`: `6,533`
- fake: `5,643`
- real: `890`
- `faceforensics++`: `5,429`
- fake: `4,066`
- real: `1,363`
- `real-ai-videos`: `66`
- fake: `33`
- real: `33`

Latest real-vs-loader check:

- every configured dataset returned `OK`
- current on-disk totals match current loader totals exactly

## Dataloader Behavior

Core implementation:

- `data/dataloader.py`

Configured datasets:

- Images:
- `ai-generated-images-vs-real-images`
- `cifake`
- `faceforensics_frames`
- Videos:
- `celeb-df-v2`
- `faceforensics++`
- `real-ai-videos`
- Frame folders:
- discovered dynamically from `datasets/preprocessed_frames` when present
- discovered dynamically from legacy `video-frames` when present

Current file-extension rules:

- Images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`

Current scanning behavior:

- Image datasets are discovered under `datasets/images/<dataset_name>`
- Split folders are discovered dynamically from disk
- Class folders are matched against the configured label map
- Images are gathered recursively
- Video datasets are discovered under `datasets/videos/<dataset_name>`
- Flat video datasets scan configured class folders recursively
- Deep nested video datasets scan configured `fake` and `real` roots recursively
- Frame-folder datasets are discovered from optional derived roots
- `datasets/preprocessed_frames` is treated as a mirrored video tree where each leaf frame folder represents one derived sample
- `video-frames` is treated as a legacy `label/dataset/video_name/*.jpg` tree where each video frame folder represents one derived sample
- Raw discovery is used for auditing, while final train/val/test behavior is determined by the active loading protocol

## Label Semantics

Global binary convention in the code:

- `0 = fake`
- `1 = real`

Important semantic caveat:

The datasets do not all represent the same notion of fake content.

- `ai-generated-images-vs-real-images`: synthetic-vs-real image task
- `cifake`: synthetic-vs-real image task
- `celeb-df-v2`: manipulated celebrity face video task
- `faceforensics++`: manipulated face video task
- `real-ai-videos`: small mixed real-vs-AI/fake video task

If combined without careful protocol design, the model can learn dataset identity or generation style rather than the intended forensic concept.

## Loader Modes and Read Path

Supported loader modes:

- `mode=\"single\"`
- `mode=\"sequence\"`

`single` mode:

- images: open with PIL and convert to RGB
- videos: sample one frame from the video
- frame folders: sample one saved frame from the frame directory

Research interpretation:

- `image_only` + `single` is the image-domain spatial path
- `video_only` + `single` is the video-domain spatial baseline
- this is not full-video ingestion; it is frame sampling from raw videos on demand

`sequence` mode:

- videos and frame folders in practice
- returns `seq_len` frames stacked as a tensor
- now uses contiguous clip sampling instead of evenly spaced sampling

Research interpretation:

- `video_only` + `sequence` is the main spatial+temporal training path
- the model still sees spatial information within each frame, while also learning temporal consistency across ordered frames
- raw videos remain the source of truth; the full video is not loaded as one giant tensor per optimization step

Video decoding path:

- if `decord` is available, use `VideoReader`
- otherwise use OpenCV

Current default sequence length:

- `seq_len = 8`

Current clip sampling behavior:

- train loader default: random contiguous clip
- validation loader default: center contiguous clip
- test loader default: center contiguous clip

Frame-folder loading behavior:

- a frame folder is treated as one derived sample corresponding to one source video
- `dtype="frame", mode="single"` samples one saved frame from the folder
- `dtype="frame", mode="sequence"` loads a contiguous subsequence of saved frames from the folder
- if the folder has fewer than `seq_len` saved frames, the last available frame is repeated to pad the sequence length
- the same train-random / eval-center contiguous policy is used for frame folders as for raw videos

Meaning:

- training sees different temporal windows from the same video across epochs
- evaluation is more stable because it uses a deterministic center clip
- this is appropriate for temporal models and still compatible with sequence-based spatial-temporal backbones
- this is also why full offline frame materialization is not required for the main raw-video training path

Current image transforms:

- Train:
- `RandomResizedCrop(224, scale=(0.9, 1.0))`
- `RandomHorizontalFlip()`
- `ColorJitter(0.2, 0.2, 0.2)`
- `ToTensor()`
- ImageNet normalization
- Validation/Test:
- `Resize((224, 224))`
- `ToTensor()`
- ImageNet normalization

Transform rationale:

- `RandomResizedCrop(224, scale=(0.9, 1.0))`
  - injects moderate spatial variation while staying close to the source framing
  - reduces brittle dependence on exact crop position
- `RandomHorizontalFlip()`
  - is acceptable for face-centric imagery where left-right orientation is usually not the target forensic cue
- `ColorJitter(0.2, 0.2, 0.2)`
  - reduces reliance on narrow brightness or color signatures
- ImageNet normalization
  - keeps preprocessing aligned with pretrained timm backbone expectations
- deterministic validation/test resize
  - keeps evaluation behavior stable and reproducible

Research implication:

- the baseline augmentation policy is intentionally moderate
- it is meant to regularize the model without making augmentation itself the main experimental driver

## Split Policy

Current split logic is protocol-aware rather than one pooled default split.

Available protocols:

- `image_only`
- `video_only`
- `frame_only`
- `combined_aux`

Default protocol resolution:

- `dtype="image"` -> `image_only`
- `dtype="video"` -> `video_only`
- `dtype="frame"` -> `frame_only`
- `dtype=None` -> `combined_aux`

### `image_only`

What it does:

- keeps image experiments separate from video experiments
- preserves source `test` boundaries for image datasets that already provide them
- derives validation from the source `train` split when no explicit validation split exists

Implication:

- image training no longer pools `train` and `test` together by default
- source image dataset boundaries are respected
- this is the primary spatial benchmark family for the image domain

### `video_only`

What it does:

- keeps video experiments separate from image experiments
- splits each video dataset independently before combining them
- uses identity-aware grouping for train/val/test assignment

Identity grouping behavior:

- Celeb-DF style names like `id0_id1_0000.mp4` are grouped by `id0`
- FaceForensics++ style names like `000_001.mp4` are grouped by `000`
- other files fall back to full path as identity

Implication:

- video split is identity-aware where naming conventions support it
- cross-dataset pooling is no longer the default behavior
- default split target is now `70/10/20` for `train/val/test`
- `mode="single"` under `video_only` is the spatial baseline on raw videos
- `mode="sequence"` under `video_only` is the spatial+temporal path on raw videos

### `frame_only`

What it does:

- keeps extracted frame-folder experiments separate from raw image and raw video experiments
- treats each frame folder as one derived sample corresponding to one original video
- uses identity-aware grouping based on the frame-folder name

Implication:

- frame-folder experiments can be run for spatial or temporal learning without pooling them into raw video runs by default
- contiguous saved-frame sequences can be loaded from derived frame directories with the same train-random / eval-center policy
- any paper results on `frame_only` should clearly state that these are derived samples materialized from the raw video corpora rather than native benchmark splits
- default split target is now `70/10/20` for `train/val/test`

### `combined_aux`

What it does:

- allows mixed training across selected datasets
- should be treated as an auxiliary experiment mode only

Implication:

- use this only when the experiment explicitly intends mixed-domain training

## Balancing Behavior

The current training split can be balanced with oversampling:

- minority-class samples are duplicated
- remaining deficit is filled by random sampling within class
- this is done in-memory before dataloader construction

Important note:

- balancing is implemented through manual oversampling inside the builder

Recommended interpretation:

- apply balancing only after choosing a clean protocol such as `image_only` or `video_only`
- `frame_only` follows the same recommendation when derived frame folders are present
- do not use balancing as a substitute for fixing dataset boundaries
- for fake-heavy video datasets, class balancing after identity-aware splitting is appropriate
- image train/val/test should follow preserved source boundaries plus derived validation from train
- video train/val/test should follow identity-aware `70/10/20`

Recommended primary experiment family:

- separate image-only models for image datasets
- separate video-only spatial baselines using `mode="single"` on raw video datasets
- separate video-only spatial+temporal models using `mode="sequence"` on raw video datasets
- optional separate frame-only models for derived frame folders
- multimodel or mixed-media runs should be reported only as auxiliary experiments unless a paper section explicitly motivates them

Why balancing is delayed until after protocol selection:

- balancing before protocol separation can hide structural mistakes
- a balanced but leaky split is still methodologically weak
- split correctness is therefore treated as the first constraint and class rebalance as the second

What the current oversampling does well:

- equalizes class counts in the training partition
- leaves validation and test untouched
- avoids conflating train-time balancing with evaluation-time corpus composition

What it does not solve:

- source leakage
- dataset-identity shortcuts
- semantic mismatch across different notions of `fake`

## Validation Tooling

### `python -m data.dataset_analyzer`

Purpose:

- compute true on-disk counts from filesystem traversal
- compute dataloader-discovered counts from `DatasetBuilder`
- compare them dataset-by-dataset

Expected result at the current project state:

- every configured dataset should show `OK` in the real-vs-loader comparison

### `python -m train.image.simulate_image_train`

Purpose:

- validate the active image-only pipeline before expensive training
- run one full train epoch and full validation/test passes on raw image datasets

Current verified status:

- completed successfully on both image datasets
- confirmed current loader correctness for:
  - `cifake`
  - `ai-generated-images-vs-real-images`

### `python -m train.image.run_image_vit`

Purpose:

- run the currently implemented ViT trainer in the new `train/` tree
- execute `IMG-EXP-01..03` with the active `image_only` split policy

Current verified status:

- `IMG-EXP-01` with `image_combined` has completed successfully
- the run directory stores checkpoints and split/config summaries in the new family-based layout
- artifact completeness should still be checked per run directory, because not every saved historical run folder under `train/image/` currently has the exact same summary-file set

### `python -m data.image_video_frame`

Purpose:

- audit images, videos, and extracted frame folders together
- compare real filesystem counts with `DatasetBuilder` discovery across all supported media types
- report frame-folder counts and total saved-frame mass when derived frame datasets are present
- optionally smoke-test torch dataset loading for image, video, and frame media

Recommended usage:

- mixed-media stats:
- `python -m data.image_video_frame`
- mixed-media validation:
- `python -m data.image_video_frame --validate`
- mixed-media loader smoke test:
- `python -m data.image_video_frame --smoke-loader`

### `python -m proc.pre_process_videos`

Purpose:

- materialize derived frame folders from the videos discovered by `DatasetBuilder`
- keep preprocessing aligned with the active dataloader instead of maintaining a separate scan path
- support rerun-safe extraction after interruption

Current semantics:

- `--frame-stride N` means keep every `N`th frame before any optional capping
- `--max-frames N` means keep at most `N` total frames per video, distributed across the full clip
- `--frame-stride 1 --max-frames None` means all frames from each video
- this is total frames per video, not frames per minute

Continuation behavior:

- if a frame folder already exists and its manifest matches the current source video and extraction settings, the folder is skipped
- if the folder is partial, missing its manifest, has a stale manifest, or reflects an older incompatible extraction shape, it is refreshed automatically
- rerun safety is therefore at video-folder granularity rather than at individual-frame granularity

Recommended usage:

- quick smoke run:
- `python -m proc.pre_process_videos --datasets celeb-df-v2 --max-frames 32 --limit 100`
- training-oriented derived frame materialization:
- `python -m proc.pre_process_videos --datasets celeb-df-v2 --max-frames 64`
- full-corpus bounded extraction:
- `python -m proc.pre_process_videos --max-frames 64`

Important note:

- full-corpus all-frame extraction is a large materialization job and should not be compared to lightweight audit commands such as `dataset_analyzer` or `dataset_run`

### `python -m data.dataset_run`

Purpose:

- enumerate every raw image and raw video sample in the main training path
- try to read every image with PIL
- try to open and probe every video with OpenCV
- surface PIL errors, metadata failures, and unreadable video/frame issues before training

Current validation checks:

- image verification using `Image.verify()`
- image reopen and RGB conversion
- video open check with `cv2.VideoCapture`
- video frame-count metadata check
- video probe reads across multiple frame indices
- frame-folder validation is intentionally separated into `data.image_video_frame`

Observed result from the validation run before deletion:

- command: `python -m data.dataset_run --num-workers 16`
- total samples checked: `192,028`
- readable: `192,016`
- failed: `12`
- failed images: `12`
- failed videos: `0`
- all `12` failures were in `datasets/images/ai-generated-images-vs-real-images`
- observed error types:
- `image file is truncated`
- `Truncated File Read`
- `broken data stream when reading image file`

Interpretation of the latest run:

- there is no current evidence of a `moov` or video metadata problem in the checked `12,028` videos
- the failures found in that run were limited to corrupted images in one image dataset
- those corrupted image files were removed in the subsequent cleanup step
- the cleaned dataset baseline should no longer be expected to hit those same PIL errors unless new bad files appear

Observed result from the cleanup run:

- command: `python -m data.dataset_fix --action delete --num-workers 16`
- unreadable images deleted successfully: `12`
- failed delete operations: `0`
- expected current dataset total after cleanup: `192,016`

Recommended confirmation sequence:

- run `python -m data.dataset_fix --action report --num-workers 16` only when checking for new failures
- run `python -m data.dataset_analyzer` after any cleanup step
- copy the latest analyzer output into this document as the post-cleanup ground truth

Recommended usage:

- full audit:
- `python -m data.dataset_run --num-workers 16`
- images only:
- `python -m data.dataset_run --dtype image --num-workers 16`
- videos only:
- `python -m data.dataset_run --dtype video --num-workers 8`
- stop at first failure:
- `python -m data.dataset_run --fail-fast --num-workers 8`

### `python -m data.dataset_fix`

Purpose:

- run the same validation pass as `dataset_run`
- then either report, delete, or quarantine unreadable files

Recommended usage:

- dry run:
- `python -m data.dataset_fix --action report --num-workers 16`
- quarantine bad files:
- `python -m data.dataset_fix --action quarantine --num-workers 16`
- delete bad files:
- `python -m data.dataset_fix --action delete --num-workers 16`

Recommendation:

- prefer `quarantine` first for reproducibility and safety
- use `delete` only after the quarantined set has been reviewed

## Preprocessing and Cleanup Script Inventory

### Active scripts

- `data/dataloader.py`
- `data/dataset_analyzer.py`
- `data/dataset_run.py`
- `data/dataset_fix.py`
- `data/video_frame_stats.py`
- `data/image_video_frame.py`
- `proc/pre_process_videos.py` as the optional active frame-materialization utility

### Legacy scripts

- everything under `datasets/temp/`
- these are historical helpers retained for reference and reproducibility context
- they should not be treated as the primary active data pipeline in current experiments

### `data/dataloader.py`

- unified dataset loader
- scans configured datasets
- scans optional derived frame-folder roots when present
- performs protocol-aware split
- applies transforms
- supports single-frame and sequence loading
- image tasks preserve source train/test boundaries
- video tasks use identity-aware splitting
- frame-folder tasks use identity-aware splitting
- sequence mode now uses contiguous clips

### `data/dataset_analyzer.py`

- real-vs-loader stats auditor
- confirms dataset coverage
- should be run whenever datasets are added, removed, or renamed

### `data/dataset_run.py`

- raw readability validator
- intended to catch corrupted images, PIL issues, OpenCV open failures, and metadata problems before training

### `data/image_video_frame.py`

- mixed-media analyzer for images, videos, and derived frame folders
- compares filesystem counts with loader discovery across media types
- can validate raw readability and smoke-test torch dataset loading

### `datasets/temp/extract_ff_frames.py` (Old / Legacy)

- extracts `5` frames per FaceForensics++ video
- performs identity-based train/val/test split
- writes derived image dataset to `datasets/images/faceforensics_frames`

### `datasets/temp/frame_converter.py` (Old / Legacy)

- legacy video-to-frame conversion utility
- infers label from path text
- writes to `video-frames`
- supported by the current dataloader when the `video-frames` root exists, but still considered a legacy derived-data path

### `datasets/temp/clean_ds.py` (Old / Legacy)

- scans for invalid videos
- can delete videos that fail metadata/frame-read checks

### `datasets/temp/del_corrupted.py` (Old / Legacy)

- scans for corrupted videos with OpenCV open/frame-count checks
- can delete unreadable video files interactively

### `datasets/temp/rgba_clean.py` (Old / Legacy)

- scans images under `datasets/`
- converts non-RGB images to RGB
- can optionally delete unreadable images

### `datasets/temp/remove_audio.py` (Old / Legacy)

- despite the name, current code is a frame extraction utility
- writes sampled frames to `frames/`

### `proc/pre_process_videos.py`

- optional active frame-materialization utility for full-video frame extraction
- no longer hard-codes a synthetic `16`-frame representation
- now records real source frame counts and supports explicit `--frame-stride` or `--max-frames`
- now discovers videos through `DatasetBuilder`
- now refreshes partial or stale frame folders automatically on rerun
- writes derived frame folders to `datasets/preprocessed_frames`
- should be treated as auxiliary preprocessing, not as a required step before video model training

### `datasets/temp/videos/faceforensics++/download.py` (Old / Legacy)

- multi-threaded FaceForensics++ downloader
- supports dataset choice, compression type, subset count, and mirror selection

## Known Risks For Research Validity

### Original benchmark split preservation

Current state:

- image datasets are scanned from disk across their source split folders
- under the active `image_only` protocol, source `test` boundaries are preserved
- validation is derived from the source `train` split only when no explicit `val` split exists

Effect:

- benchmark comparability is stronger than the older pooled pipeline because source test sets are no longer mixed back into training
- any reported results should still state when validation was derived from the source training split instead of using an official source validation split

### Source leakage risk

Current issue:

- image datasets often fall back to full-path identity
- no robust source-grouping logic exists for many image datasets

Effect:

- near-duplicate or source-correlated samples can cross splits

### Cross-dataset shortcut learning

Current issue:

- image-generation datasets and face-manipulation video datasets are pooled under one binary label scheme

Effect:

- model may learn dataset-specific artifacts rather than forensic evidence

### Severe size imbalance

Observed imbalance:

- `cifake` dominates the image pool
- `celeb-df-v2` is fake-heavy
- `real-ai-videos` is very small

Effect:

- apparent performance can be driven by dominant datasets and not generalization

Current interpretation from the latest video frame statistics:

- file-level imbalance in the main video datasets is still substantial
- frame-mass is also fake-dominant for `celeb-df-v2`
- frame-mass is also fake-dominant for `faceforensics++`
- `real-ai-videos` is real-dominant by total frames and duration, but it is too small to rebalance the main video corpus
- therefore balancing and protocol separation are still required during training; frame-mass alone does not neutralize the class skew in the main video experiments
- the presence of many raw frames in FF++ should not be interpreted as a reason to materialize the full corpus into JPEG folders before training

### Mixed semantic definition of “fake”

Current issue:

- “fake” includes both fully synthesized content and manipulated real media

Effect:

- the paper must define exactly what target concept is being modeled in each experiment

### Derived-frame interpretation risk

Current issue:

- extracted frame folders are not native benchmark datasets
- they are materialized views of the raw video corpora

Effect:

- any frame-only result must be labeled as a derived-video experiment
- otherwise the provenance and split semantics become ambiguous

### Training-control comparability risk

Current issue:

- model-family comparisons can become noisy if optimizer and scheduler policy drift between families

Current mitigation:

- the implemented image branches currently use one shared optimization stack

Effect:

- backbone comparisons are more defensible in the present phase
- later optimizer ablations should be reported explicitly as a separate stage

## Required Logging For Each Experiment

For every training run, record:

- date and git commit
- exact dataset list used
- exact counts per dataset and class
- whether files were validated with `data.dataset_run`
- dataloader mode: `single` or `sequence`
- original splits preserved or rebuilt
- split ratios
- identity grouping logic used
- balancing enabled or disabled
- augmentation policy
- image size and sequence length
- number of unreadable files found and removed
- training/validation/test counts after all filtering
- evaluation protocol and checkpoint selection rule

## Current Interpretation

The current repository has a verified and internally consistent dataset inventory. Before image cleanup, the analyzer confirmed that on-disk totals matched loader totals exactly. The expected post-cleanup baseline is `192,016` discoverable samples with `179,988` images and `12,028` videos.

On the current checkout, no derived frame-folder roots are present yet. The dataloader and mixed-media analyzer now support them, but they are not part of the current counted baseline until those roots are generated.

The bigger research problem is no longer basic dataset discovery. The real concerns are:

- image source leakage risk where datasets do not provide stronger identity grouping
- custom validation derivation for image datasets that only ship `train` and `test`
- possible source leakage in image data
- cross-dataset shortcut learning
- inconsistent semantics of the fake class

Those points should drive the redesign of the training protocol before new experiments are run.
