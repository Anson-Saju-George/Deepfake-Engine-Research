# Data Collection

## Active Sources

### Images

- `cifake`
- `ai-generated-images-vs-real-images`

### Videos

- `celeb-df-v2`
- `faceforensics++`
- `real-ai-videos`

## Current On-Disk Tree Summary

Verified top-level datasets currently present:

- `datasets/images/ai-generated-images-vs-real-images`
- `datasets/images/cifake`
- `datasets/videos/celeb-df-v2`
- `datasets/videos/faceforensics++`
- `datasets/videos/real-ai-videos`

Optional or derived roots that may appear depending on preprocessing state:

- `datasets/preprocessed_frames`
- `datasets/images/faceforensics_frames`
- `video-frames`
- `frames`

## Current Dataset Snapshot

Latest cleaned raw dataset state:

- total raw samples: `192,016`
- images: `179,988`
- videos: `12,028`
- labels: `99,740 fake`, `92,276 real`

Current verified image datasets:

- `ai-generated-images-vs-real-images`: `59,988`
- `cifake`: `120,000`

Current verified video datasets:

- `celeb-df-v2`: `6,533`
- `faceforensics++`: `5,429`
- `real-ai-videos`: `66`

## Dataset Structure Notes

### `ai-generated-images-vs-real-images`

- structure: `train/{fake,real}` and `test/{fake,real}`
- latest counts:
- `train/fake`: `23,998`
- `train/real`: `23,993`
- `test/fake`: `6,000`
- `test/real`: `5,997`

### `cifake`

- structure: `train/{FAKE,REAL}` and `test/{FAKE,REAL}`
- counts:
- `train/FAKE`: `50,000`
- `train/REAL`: `50,000`
- `test/FAKE`: `10,000`
- `test/REAL`: `10,000`

### `celeb-df-v2`

- structure: flat class folders
- `Celeb-fake`: `5,643`
- `Celeb-real`: `590`
- `YouTube-real`: `300`

### `faceforensics++`

- structure used by loader: nested `fake/**` and `real/**`
- fake total: `4,066`
- real total: `1,363`

### `real-ai-videos`

- structure: flat class folders
- `fake`: `33`
- `real`: `33`

## Collection Structure

The repo expects datasets under `datasets/` with modality-specific subtrees:

- `datasets/images/...`
- `datasets/videos/...`

Derived frame folders may also exist under:

- `datasets/preprocessed_frames/...`
- `video-frames/...`

## Collection Principles

- raw dataset layout is preserved where possible
- source dataset naming conventions are used for class and identity inference
- derived frame folders are treated as secondary artifacts, not as original source corpora

## Label Semantics

Global binary convention in code and docs:

- `0 = fake`
- `1 = real`

Important semantic caveat:

- image datasets here are largely synthetic-vs-real tasks
- video datasets here are largely manipulation-vs-real tasks

These should not be casually collapsed into one core benchmark claim without protocol-level separation.
