# Video Experiment Config

This file is the working video experiment register for the new top-level `train/` tree.

The goal is to keep the video plan explicit before implementation starts, and to preserve every planned run in the research record even if a model later underperforms or fails.

Current training command sheet:

- `train/commands.md`

## Global Video Training Policy

- protocol: `video_only`
- source of truth: raw videos
- offline frame materialization required: `no`
- default epochs: `10`
- balancing: enabled after dataset scope and split policy are correct
- split strategy:
  - each raw-video dataset is split independently
  - split policy is identity-aware `70/10/20` for `train/val/test`
- clip sampling:
  - train: random contiguous clip
  - eval: center contiguous clip
- core dataset scopes:
  - `celeb-df-v2`
  - `faceforensics++`
  - combined raw-video setting using both
- research target:
  - establish strong spatial and temporal baselines first
  - only then judge the higher-complexity spatial+temporal hybrids

## Core Category Logic

The video plan is organized into three experiment categories:

### Spatial

- raw-video input source
- loader behavior: `dtype="video"`, `mode="single"`, `seq_len=1`
- interpretation:
  - one sampled frame per selected raw video
  - no explicit temporal modeling
  - still a valid video-domain baseline because the source is the video dataset

Why this matters:

- it measures what can be learned from frame-level forensic cues alone
- it gives a serious baseline before claiming temporal models add value

### Temporal

- raw-video input source
- loader behavior: `dtype="video"`, `mode="sequence"`
- interpretation:
  - contiguous ordered clips sampled on demand from raw videos
  - emphasizes temporal consistency, motion, and cross-frame artifacts

Why this matters:

- it isolates the value of temporal structure over the spatial baseline

### Spatial+Temporal

- raw-video input source
- loader behavior: `dtype="video"`, `mode="sequence"`
- interpretation:
  - strong per-frame spatial reasoning plus richer clip-level aggregation or selection

Why this matters:

- it is the final richer category once the plain spatial and plain temporal baselines are understood

## Raw-Video Policy

- main video training should read raw videos directly
- the dataloader samples frames or contiguous clips on demand
- full offline frame dumping is not required for the main video path
- extracted frame folders remain auxiliary only

## Experiment Register

### VID-SPA-01

- experiment number: `VID-SPA-01`
- category: `spatial`
- family: `ConvNeXt`
- model name: `convnext_base`
- mode: `single`
- seq_len: `1`
- batch size: `16`
- epochs: `10`
- dataset scopes:
  - Celeb-DF-only
  - FaceForensics++-only
  - combined raw-video setting
- why we use it:
  - primary raw-video spatial baseline
  - uses a strong modern CNN already trusted in the legacy tree
- what it tells us:
  - whether frame-level forensic cues from raw videos are already strong before sequence modeling
- legacy reference:
  - `video_celebdf`
  - `video_ffpp`
  - `video_combined`
- status: `planned`

### VID-SPA-02

- experiment number: `VID-SPA-02`
- category: `spatial`
- family: `Xception`
- model name: `xception71`
- mode: `single`
- seq_len: `1`
- batch size: `16`
- epochs: `10`
- dataset scopes:
  - Celeb-DF-only
  - FaceForensics++-only
  - combined raw-video setting
- why we use it:
  - strong forensic-style CNN family
  - gives a second spatial baseline with a different backbone bias
- what it tells us:
  - whether the backbone family materially changes frame-level sensitivity on raw videos
- legacy reference:
  - `baseline_xception`
  - `combined_xception`
- status: `planned`

### VID-TMP-01

- experiment number: `VID-TMP-01`
- category: `temporal`
- family: `ConvNeXt Sequence`
- model name: `convnext_base`
- mode: `sequence`
- seq_len: `8`
- batch size: `8`
- epochs: `10`
- dataset scopes:
  - Celeb-DF-only
  - FaceForensics++-only
  - combined raw-video setting
- why we use it:
  - primary contiguous-clip temporal baseline
  - simplest clean test of temporal information value
- what it tells us:
  - whether ordered clip structure improves over the spatial baselines
- legacy reference:
  - `video_celebdf`
  - `video_ffpp`
  - `video_combined`
- status: `planned`

### VID-TMP-02

- experiment number: `VID-TMP-02`
- category: `temporal`
- family: `ResNet+LSTM`
- model name: `resnet50.a1_in1k`
- mode: `sequence`
- seq_len: `16`
- batch size: `8`
- epochs: `10`
- dataset scopes:
  - Celeb-DF-only
  - FaceForensics++-only
  - combined raw-video setting
- why we use it:
  - explicit temporal aggregation model
  - tests whether a dedicated recurrent head helps over simpler clip pooling
- what it tells us:
  - whether temporal head design matters beyond just giving the model a clip
- legacy reference:
  - `resnet_lstm`
- status: `planned`

### VID-ST-01

- experiment number: `VID-ST-01`
- category: `spatial_temporal`
- family: `ConvNeXt Hybrid`
- model name: `convnext_base`
- mode: `sequence`
- seq_len: `12`
- batch size: `8`
- epochs: `10`
- dataset scopes:
  - Celeb-DF-only
  - FaceForensics++-only
  - combined raw-video setting
- why we use it:
  - richer clip model combining strong frame reasoning with more expressive clip-level aggregation
- what it tells us:
  - whether hybrid aggregation beats the simpler temporal baselines
- legacy reference:
  - `topk_convnext`
  - `freq_convnext`
  - `combined_convnext`
- status: `planned`

### VID-ST-02

- experiment number: `VID-ST-02`
- category: `spatial_temporal`
- family: `Xception Hybrid`
- model name: `xception71`
- mode: `sequence`
- seq_len: `12`
- batch size: `8`
- epochs: `10`
- dataset scopes:
  - Celeb-DF-only
  - FaceForensics++-only
  - combined raw-video setting
- why we use it:
  - hybrid forensic-style clip model using xception71 as the frame backbone
- what it tells us:
  - whether xception-style frame reasoning benefits from richer temporal aggregation
- legacy reference:
  - `topk_xception`
  - `freq_xception`
  - `combined_xception`
- status: `planned`

## Logging Rules

For every video experiment, keep the entry even if the run fails.

Minimum fields to append later for each run:

- actual dataset scope
- actual trainer entrypoint
- seed
- mode
- seq_len
- batch size
- sampling policy
- best validation score
- final test score
- failure mode if unsuccessful
- notes on what changed from the original plan

## Current Intent

This markdown file is mirrored in `train/video_model.py`. They should remain aligned:

- same experiment numbers
- same category meanings
- same defaults
- same rationale

Current implementation note:

- this file is currently a planning and registry document
- actual migrated video training execution code in the new `train/` tree has not been implemented yet
- the image side is currently ahead of the video side in implementation
