# Exploratory Data Analysis (EDA)

## Core Questions

- What is the class distribution by modality and dataset?
- How imbalanced are the video datasets?
- Does frame mass materially change the interpretation of video-level imbalance?
- Are on-disk counts consistent with dataloader discovery?

## Current Findings

Raw cleaned dataset totals:

- total: `192,016`
- images: `179,988`
- videos: `12,028`
- fake: `99,740`
- real: `92,276`

Image dataset findings:

- `ai-generated-images-vs-real-images`
- total: `59,988`
- fake: `29,998`
- real: `29,990`

- `cifake`
- total: `120,000`
- fake: `60,000`
- real: `60,000`

Video dataset findings:

- `celeb-df-v2`
- total: `6,533`
- fake: `5,643`
- real: `890`

- `faceforensics++`
- total: `5,429`
- fake: `4,066`
- real: `1,363`

- `real-ai-videos`
- total: `66`
- fake: `33`
- real: `33`

Video observations:

- `celeb-df-v2` is strongly fake-heavy by video count
- `faceforensics++` is fake-heavy by video count
- `real-ai-videos` is tiny and should be treated cautiously

Frame-mass interpretation:

- frame-mass analysis adds nuance
- it does not fully remove the core imbalance in the main video benchmarks

Current interpretation from video frame statistics:

- `celeb-df-v2` remains fake-dominant by total frame mass and total duration
- `faceforensics++` remains fake-dominant by total frame mass and total duration
- `real-ai-videos` is real-dominant by frame mass and duration, but too small to rebalance the main corpus

Methodological implication:

- protocol separation and post-split balancing remain necessary
- frame-mass alone should not be used as an argument that the main video corpus is effectively balanced
- large raw frame mass in FF++ should not be interpreted as a reason to materialize all frames before training; raw-video sequence loading is the preferred main path

## How To Read These Numbers Correctly

The EDA in this repo separates several different notions of distribution.

File-count distribution:

- how many image files or video files belong to each class
- useful for understanding sample-count imbalance

Frame-mass distribution:

- how many raw frames exist behind the video files in each class
- useful for understanding whether one class systematically has longer clips

Duration distribution:

- how much real clock-time footage exists per class
- useful because a dataset can be balanced by file count while still be unbalanced by total footage

Loader-discovery distribution:

- what the active dataloader actually sees
- this is the number that matters operationally for training and dataset audit consistency

Why these should not be conflated:

- a fake-heavy dataset can remain fake-heavy even after frame-mass analysis
- a balanced-by-file dataset can still be unbalanced by duration
- a growing derived-frame corpus can make naive totals look larger than the raw benchmark inventory

## EDA Interpretation Rules Used In This Project

- raw image and raw video totals are treated as the baseline corpus truth
- derived frame folders are reported separately from the raw corpus
- image datasets are interpreted as image-domain spatial benchmarks
- video datasets are interpreted separately as video-domain spatial and spatial+temporal benchmarks
- frame mass is used to contextualize skew, not to rhetorically erase it

## Why EDA Matters For The Paper

The EDA section is not cosmetic in this project. It determines what claims are defensible.

It supports:

- whether class balancing is needed
- whether source train/test boundaries are being respected
- whether temporal models are being evaluated on an honestly skewed corpus
- whether frame-derived experiments are being presented with the correct provenance
- whether strong results may actually be driven by dataset identity rather than forensic reasoning

## Dataloader Agreement

Current expected state:

- on-disk totals and dataloader-discovered totals should match for each configured raw dataset
- if frame extraction is in progress, total discoverable sample counts can rise because frame folders become additional `dtype="frame"` records

## Questions to Keep Answering

- Are on-disk counts still aligned with loader discovery after preprocessing changes?
- Are frame-derived counts being reported separately from raw corpora?
- Are dataset-level class skews being described honestly before balancing is applied?
- Are spatial baselines and spatial+temporal models being reported as distinct experiment families?

## Active Analysis Utilities

- `python -m data.dataset_analyzer`
- `python -m data.video_frame_stats`
- `python -m data.image_video_frame`
- `python train_data_pipeline_pull.py`
