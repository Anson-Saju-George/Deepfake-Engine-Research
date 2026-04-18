# Feature Engineering

## What This Stage Means

Feature engineering is the stage where a project decides what representation the model will learn from.

In older machine-learning systems, this often meant designing handcrafted features directly.

In modern deep learning, feature engineering often shifts into a different form:

- choosing the input representation
- choosing what transformations are applied
- choosing whether the model sees single images, single video frames, or temporal clips
- deciding whether the system should learn from raw media or from derived artifacts such as extracted frames

So even though this repo does not rely on heavy handcrafted forensic descriptors, feature engineering is still happening.

## The Actual Feature Spaces In This Repo

The project currently works with four practical representation types.

Image tensors:

- RGB tensors derived from raw image files

Single sampled video frames:

- one frame sampled from a raw video
- used for the video-domain spatial baseline

Contiguous raw-video clips:

- ordered sequences of frames sampled from raw video
- used for spatial+temporal learning

Contiguous derived-frame sequences:

- ordered saved frames from extracted frame folders
- treated as derived-video experiments rather than as the default path

## Why This Counts As Feature Engineering

The question is not only `what model are we using?`
It is also:

- what information is being exposed to the model
- what information is being suppressed
- what inductive bias is being created by the input format

Example:

- a single image input emphasizes spatial artifacts
- a sequence input adds motion and temporal consistency cues

That is a feature-engineering decision, even if the network learns the features internally.

## Spatial Versus Temporal Representation

This is one of the central design choices in the repo.

Image domain:

- purely spatial

Video `mode="single"`:

- still spatial
- but sampled from the video domain rather than from native image datasets

Video `mode="sequence"`:

- spatial within each frame
- temporal across ordered frames

This distinction is extremely important for the thesis.

It means:

- image models are not just smaller video models
- single-frame video baselines are not the same thing as temporal video models
- derived-frame experiments should not be reported as if they were identical to native raw-video sequence learning

## Transform Stack As Representation Design

The preprocessing transforms also shape the effective feature space.

Current image transform design does the following:

- keeps the main content in view
- allows moderate spatial variation
- reduces reliance on exact framing
- reduces brittle dependence on raw color or brightness quirks
- aligns tensors with pretrained backbone assumptions through ImageNet normalization

So the transform stack should be seen as a representation design choice, not just boilerplate augmentation.

## What The Repo Is Not Doing Yet

The current baseline path does not yet make heavy use of:

- handcrafted frequency-domain descriptors
- explicit compression artifact channels
- face-landmark geometry features
- optical flow features
- audio-video fusion

That is intentional.

Why:

- the current phase prioritizes strong baseline representation choices before specialized feature-engineering ablations

## Future Feature Engineering Ablations Worth Studying

These are valid future research directions, but they should be reported explicitly as ablations rather than hidden in the baseline.

Possible studies:

- face-cropped versus full-frame inputs
- larger image resolution versus standard `224x224`
- sequence length sensitivity
- frame-only versus raw-video sequence learning
- frequency-domain or compression-aware channels
- oversize-image resize-policy studies for `ai-generated-images-vs-real-images`

## What The Thesis Or Paper Should State

The write-up should explain that the project uses learned representations, but that representation design is still explicit.

It should state:

- which input representations are used
- which are treated as baselines
- which are treated as temporal models
- which are derived rather than native
- which feature-engineering ideas were deliberately postponed for later ablation studies
