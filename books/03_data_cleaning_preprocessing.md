# Data Cleaning & Preprocessing

## What This Stage Means

Cleaning and preprocessing are the stages where the raw corpus is checked for readability, corrected where possible, and transformed into a form that the training pipeline can use consistently.

In this repository, this stage is especially important because:

- corrupted files existed in the collected data
- video preprocessing can generate large derived artifacts
- weak preprocessing choices can quietly change the meaning of the experiment

## Cleaning History In This Repo

Two historical cleanup facts matter and should remain part of the permanent record.

Image cleanup:

- `12` corrupted images were removed from `ai-generated-images-vs-real-images`

Historical video cleanup:

- `3` bad videos had already been deleted earlier due to old `moov`-type or metadata issues

Pre-cleanup analyzer state:

- total raw samples: `192,028`

Post-cleanup analyzer state:

- total raw samples: `192,016`

These numbers matter because they show that the current benchmark inventory is the result of an explicit cleaning process, not just whatever happened to remain on disk.

## Why Cleaning Matters Here

A corrupted sample is not just a nuisance. It creates several risks:

- loader crashes during training
- inconsistent dataset counts between disk and dataloader discovery
- unstable preprocessing runs
- misleading reporting if bad files are silently skipped instead of accounted for

The repo therefore treats cleaning as a documented research step, not as an invisible one-off fix.

## Active Cleaning And Validation Tools

The active tools are:

- `python -m data.dataset_run`
- `python -m data.dataset_fix`
- `python -m data.dataset_analyzer`

What they do:

- `dataset_run`
  - validates that raw files can actually be opened and read
- `dataset_fix`
  - can report, quarantine, or delete unreadable files
- `dataset_analyzer`
  - confirms final counts and agreement between filesystem truth and loader discovery

## What Preprocessing Means In This Repo

Preprocessing in this repo has two very different meanings, and they must not be confused.

Runtime preprocessing:

- transforms applied during dataloader operation
- resizing, cropping, flipping, normalization, and video-frame sampling

Materialization preprocessing:

- writing new derived artifacts to disk
- mainly frame-folder extraction from raw videos

The first is part of normal training behavior.
The second is optional and should be reported separately.

## Active Preprocessing Path

The accepted active extractor is:

- `proc/pre_process_videos.py`

What it does:

- discovers videos through the active `DatasetBuilder`
- writes frame folders in a continuation-safe way
- records extraction manifests
- supports both bounded extraction and all-frame extraction

Why this matters:

- the extractor now follows the same discovery logic as the dataloader
- preprocessing therefore stays aligned with the actual training corpus

## Critical Preprocessing Semantics

These terms must be understood precisely.

`frame_stride=1`

- means every eligible frame is kept
- it does not mean one frame per second or one frame per minute

`max_frames=None`

- means no cap is applied after stride

`max_frames=N`

- means keep at most `N` total frames per video
- it is not a time-based cap

Continuation-safe reruns:

- complete compatible frame folders are skipped
- stale or partial frame folders are refreshed

## Why Full Frame Materialization Is Not The Default

This repo explicitly moved away from treating full frame extraction as mandatory.

Why:

- raw-video sequence loading is already supported
- full-frame materialization can consume huge storage
- adjacent video frames are highly redundant
- large frame stores create additional maintenance burden

Research implication:

- raw-video training is the primary video path
- frame extraction is an auxiliary branch for derived-frame experiments

## Rejected Or Deferred Routes

The repo evaluated hardware-accelerated extraction paths and did not adopt them as the active baseline.

Why this matters:

- rejected paths are part of the methodological record
- they explain why the project stayed with the CPU/dataloader-aligned extractor

Current accepted conclusion:

- CPU extraction remains the accepted baseline preprocessing route

## Legacy Utilities And Why They Matter

Old helper scripts were moved into `datasets/temp/`.

They still matter historically because they show:

- earlier extraction strategies
- earlier cleaning strategies
- legacy dataset transformations

But they should not be presented as the current canonical workflow.

## What The Thesis Or Paper Should State

The final write-up should clearly state:

- what was cleaned
- how many files were removed
- that raw corpus totals changed after cleaning
- what preprocessing is part of runtime loading
- what preprocessing creates derived artifacts on disk
- that frame extraction is optional in the current main methodology
