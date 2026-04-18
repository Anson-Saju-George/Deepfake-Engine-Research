# Data Collection

## What This Stage Means

Data collection is the stage where the project defines what raw material exists, what form it takes, and what those datasets actually represent.

In a research project, this stage is not just an inventory list. It is where the study defines:

- what the model is allowed to learn from
- how broad or narrow the target domain is
- how comparable the datasets are
- what semantic caveats will follow the work into training and evaluation

## What Is Collected In This Repo

The repository currently uses two image datasets and three video datasets as the cleaned raw baseline.

Image datasets:

- `cifake`
- `ai-generated-images-vs-real-images`

Video datasets:

- `celeb-df-v2`
- `faceforensics++`
- `real-ai-videos`

The active on-disk roots are:

- `datasets/images/...`
- `datasets/videos/...`

Optional derived roots may also appear later:

- `datasets/preprocessed_frames`
- `datasets/images/faceforensics_frames`
- `video-frames`
- `frames`

These derived roots are not part of the core raw dataset baseline unless explicitly generated and reported.

## Current Raw Dataset Snapshot

Latest cleaned raw totals:

- total raw samples: `192,016`
- images: `179,988`
- videos: `12,028`
- labels: `99,740 fake`, `92,276 real`

Image side:

- `ai-generated-images-vs-real-images`: `59,988`
- `cifake`: `120,000`

Video side:

- `celeb-df-v2`: `6,533`
- `faceforensics++`: `5,429`
- `real-ai-videos`: `66`

## Why These Datasets Matter

### `cifake`

What it is:

- a large image benchmark with a clean fake/real structure

Why it matters:

- it provides a strong image-domain baseline
- it is large enough to support stable architecture comparison
- it is balanced enough to reduce confusion between model weakness and class skew

### `ai-generated-images-vs-real-images`

What it is:

- another image-domain synthetic-vs-real benchmark with a different corpus profile

Why it matters:

- it tests generalization beyond one image dataset
- it helps reveal whether a model is learning transferable image cues or only dataset-specific artifacts

### `celeb-df-v2`

What it is:

- a face-manipulation video dataset with clear fake-heavy class skew

Why it matters:

- it is one of the main raw-video benchmarks in the repo
- it stresses identity-aware splitting and honest handling of imbalance

### `faceforensics++`

What it is:

- a major manipulated-face video corpus with nested structure and substantial clip length

Why it matters:

- it provides a second major raw-video benchmark
- it is important for studying both spatial and spatial+temporal learning on manipulated videos

### `real-ai-videos`

What it is:

- a very small video dataset with balanced file counts

Why it matters:

- it can support auxiliary experiments
- it should not be allowed to dominate conclusions because it is too small to stabilize the overall video story

## Why Collection Cannot Be Treated As Neutral

Deepfake datasets are not semantically identical.

Image-side `fake` often means:

- fully synthetic imagery

Video-side `fake` often means:

- manipulated real footage
- face swaps
- other temporal manipulation artifacts

This means a single binary label does not guarantee one unified forensic concept.

The collection stage therefore already creates an important research caveat:

- `fake` is not perfectly homogeneous across all datasets in the repo

## Structure Of The Collected Data

Image datasets:

- generally use source split folders such as `train` and `test`
- are consumed as image-domain spatial tasks

Video datasets:

- may use flat class folders
- may use deeper nested structures
- are consumed as raw video rather than as fully materialized frame corpora by default

Derived frame folders:

- when present, they are secondary artifacts
- they are not the same thing as the original raw datasets

## Collection Principles Used In This Project

The repo follows these collection principles:

- preserve raw dataset layout where possible
- keep modality roots separate
- do not redefine image and video as one default corpus
- treat derived frame folders as downstream artifacts, not as original source data
- document label semantics honestly

## Collection Risks That Must Be Admitted

The collection stage already contains research risks.

Source-style bias:

- a model can learn dataset style instead of forensic evidence

Semantic mismatch:

- `fake` may mean synthetic generation in one dataset and manipulation in another

Dataset dominance:

- larger datasets can dominate combined training behavior

Small auxiliary-set fragility:

- tiny datasets can look informative but are statistically weak on their own

## What The Thesis Or Paper Should State

The final write-up should clearly state:

- which datasets are image datasets and which are video datasets
- how many cleaned raw samples exist
- that derived frame folders are optional and secondary
- that label semantics differ across dataset families
- that the repository intentionally separates image and video protocols to respect those differences
