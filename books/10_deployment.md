# Deployment

## What This Stage Means

Deployment is the stage where a trained model stops being only a research artifact and becomes an inference system that another person, service, or workflow can actually use.

In other words, deployment answers the question:

- how does a model leave the training notebook or training script and become something that can accept new unseen data and return a prediction in a controlled, repeatable way?

For people new to the field, this distinction matters.
A trained checkpoint by itself is not yet a deployed system.
A checkpoint is only a saved set of learned weights.
Deployment requires turning those weights into a usable and reproducible inference path.

That path usually needs:

- an input contract
- a preprocessing contract
- a model-loading contract
- an output contract
- artifact versioning
- runtime monitoring

Without those pieces, a model may exist technically but still be unusable in practice.

## Why Deployment Is Not The First Priority In This Repo

This repository is first a research repository, not a product repository.

That means the immediate goal is not simply to serve predictions as fast as possible.
The immediate goal is to make sure that any prediction-serving path is backed by defensible experimental methodology.

That is why this project has deliberately prioritized:

- dataset truth
- protocol separation
- split correctness
- preprocessing correctness
- auditable run records
- interpretable experiment structure

This order is intentional.
If the data protocol is weak, deployment only makes a weak system easier to use.
That is not a research success.
That is just a faster way to operationalize uncertainty.

## Why Deployment Still Belongs In The Lifecycle Documentation

Even when deployment is not yet the active implementation focus, it still matters because it changes how earlier stages should be designed.

Examples:

- checkpoint naming needs to preserve provenance so later packaging is not ambiguous
- preprocessing rules need to be stable so inference uses the same assumptions as evaluation
- protocol separation matters because image inference and video inference are not the same system
- derived frame experiments must not be confused with raw-video deployment behavior

So deployment is not just a final chapter.
It influences how the earlier research stages should be documented and structured.

## What Deployment Means For This Specific Project

This project is not solving one single deployment problem.
It is solving multiple related but distinct inference problems.

### Image Deployment Problem

Input:

- one image

Output:

- fake or real prediction
- optionally a confidence score or class probability

Operational meaning:

- spatial-only inference
- simpler interface
- lower runtime complexity than video inference

### Video Spatial Baseline Deployment Problem

Input:

- one raw video

Processing behavior:

- sample one or more frames from the raw video in a way consistent with the evaluation policy

Output:

- fake or real prediction for the video

Operational meaning:

- still video-domain inference
- but not full temporal reasoning
- closer to frame-sampled video classification

### Video Spatial+Temporal Deployment Problem

Input:

- one raw video

Processing behavior:

- sample a contiguous clip from the raw video
- preserve order across frames
- send that sequence to a temporal-capable model path

Output:

- fake or real prediction for the video

Operational meaning:

- higher inference complexity
- stronger methodology for motion and temporal consistency analysis
- more sensitive to clip-sampling policy

These are not interchangeable systems.
A paper, thesis, README, or demo should never collapse them into one vague phrase such as “the deployed deepfake detector.”
That would hide important methodological differences.

## What A Correct Future Deployment Layer Must Preserve

If deployment is added later, it should preserve the same semantics used during training and evaluation.

That means the future inference layer must respect at least the following.

### Protocol Identity

A deployed system should always know whether it is serving:

- `image_only`
- `video_only`
- `frame_only`
- `combined_aux`

Why this matters:

- image preprocessing assumptions differ from raw-video decoding assumptions
- frame-derived systems are not equivalent to raw-video systems
- a combined auxiliary path should not silently become the public default

### Preprocessing Parity

Inference should apply the same essential preprocessing logic used during evaluation.

For example:

- resize behavior should match the trained backbone expectation
- normalization should match the training transform stack
- video clip extraction should match the evaluation semantics if a single deterministic prediction is required

If the preprocessing differs too much between evaluation and deployment, the deployed system is no longer measuring the same thing that the validation and test metrics described.

### Artifact Provenance

Every served model should be traceable back to:

- experiment ID
- model family
- model name
- dataset scope
- config
- checkpoint summary

This repository has already started building that traceability into the image save structure:

- `train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/`

That matters for deployment because a future service should not be loading a checkpoint whose experimental identity is unclear.

## Why Stable Save Structure Matters For Deployment

A deployable model is not just “the best file in a folder.”
A deployable model must be attributable.

The current run structure already helps future deployment because it stores:

- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`
- `best.pth`
- `last.pth`

This means a future deployment engineer can answer:

- what model is this?
- what data scope trained it?
- what split policy produced it?
- what metric selected it?
- what final test result was associated with it?

That is exactly the kind of traceability a serious research pipeline should preserve before deployment begins.

## What This Repo Should Avoid During Deployment

Several deployment mistakes would damage the clarity of this project.

### Mistake 1: Hiding Protocol Differences

Bad example:

- serving one endpoint that mixes image and video logic without making the modality explicit

Why it is bad:

- it hides what model assumptions are in play
- it makes evaluation provenance harder to explain
- it creates ambiguity about preprocessing behavior

### Mistake 2: Treating Derived Frames As Raw Video

Bad example:

- extracting frames offline and then presenting the deployed model as if it were a native raw-video system

Why it is bad:

- derived-frame inference and raw-video inference are not methodologically identical
- the runtime and sampling assumptions are different
- the paper would risk overstating what the deployed system actually does

### Mistake 3: Deploying Before Benchmark Stability

Bad example:

- productizing a model family before the dataset protocol and evaluation story are stable

Why it is bad:

- it front-loads engineering effort before the scientific claims are secure
- it makes later methodology fixes more painful because product assumptions become harder to unwind

## What A Responsible Future Deployment Stack Would Likely Need

A future deployment layer for this repository should likely include:

- explicit modality-aware entrypoints
- checkpoint registry or model manifest
- deterministic evaluation-style preprocessing mode
- optional batch or streaming inference for video
- clear output format with label and confidence
- version tagging tied to experiment IDs
- documentation stating whether the served model is image, video-spatial, or video-spatial-temporal

It may also later include:

- explainability aids
- score calibration
- threshold management by use case
- inference latency measurements
- hardware/runtime profiling

But those should come after the basic methodological contract is stable.

## What The Thesis Or Paper Should Say About Deployment

A serious thesis or journal paper should describe deployment honestly.

At the current project stage, the correct framing is:

- deployment is a downstream extension, not the present center of the work
- current effort prioritizes methodological rigor and experimental auditability
- any future inference system must preserve protocol-specific semantics and preprocessing parity
- deployment should follow validated baselines, not replace the need for them

## Current Repo Truth

At the time of writing:

- image training in the new `train/` tree is active
- video training in the new `train/` tree is also active for the current image-style video backbones
- raw-video training remains the main video philosophy
- full frame materialization is auxiliary, not the primary deployment or training assumption

Important deployment caveat:

- reserved native-video IDs are still future work
- so any future deployment discussion should distinguish:
  - active image models
  - active spatial video models
  - active clip-based video models
  - future native-video models

That means deployment discussion in this repository should remain precise and forward-looking, not overstated.
