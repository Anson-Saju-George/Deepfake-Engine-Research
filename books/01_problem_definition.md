# Problem Definition

## What This Stage Means

Problem definition is the stage where the project decides what question it is actually trying to answer.

In machine learning, this matters because a model can appear successful while solving the wrong problem. A repo can say "deepfake detection," but that phrase can hide several different tasks:

- detecting synthetic images
- detecting manipulated videos
- detecting AI-generated videos
- detecting artifacts in extracted frames

These are related, but they are not interchangeable.

## What The Core Problem Is In This Repo

This repository is trying to build a research-grade deepfake detection pipeline that is methodologically defensible.

That means the project is not only about achieving high accuracy. It is also about:

- using correct dataset boundaries
- avoiding split leakage
- separating image and video experiments by default
- documenting preprocessing truthfully
- producing experiment records that can support a thesis or journal paper

## Why The Problem Is Hard

Deepfake detection is difficult for three main reasons.

Media heterogeneity:

- an image is a static sample
- a video is a temporal sample
- an extracted frame folder is a derived representation of a video, not a native benchmark item

Dataset heterogeneity:

- some datasets represent synthetic generation
- some represent face manipulation
- some are balanced
- some are strongly fake-heavy

Methodological fragility:

- poor split choices can leak identity or source cues
- pooled multimodal training can blur what the model is really learning
- undocumented preprocessing can make experiments difficult to defend

## What The Repo Chooses To Treat As The Real Problem

The repo makes a deliberate decision:

- image and video should not be treated as the same default task

Current operational framing:

- `image_only` = image-domain spatial benchmark family
- `video_only` with `mode="single"` = video-domain spatial baseline
- `video_only` with `mode="sequence"` = video-domain spatial+temporal family
- `frame_only` = derived-video experiment family
- `combined_aux` = auxiliary only, not the default benchmark path

Important implementation note:

- the registry still separates `temporal` and `spatiotemporal` video categories
- however, the current active sequence trainer uses per-frame image-backbone encoding followed by temporal mean pooling for both
- so the present implementation truth is best read as:
  - spatial video modeling
  - clip-based video modeling

This is the most important problem-definition decision in the codebase.

## Research Objective

The objective is to produce reliable, reproducible deepfake detection experiments that can be defended technically.

In practical terms, success means:

- the dataloader reflects actual dataset truth
- image train/test boundaries are respected
- video split policy is identity-aware
- spatial and spatial+temporal video claims are separated
- every major experimental choice is documented clearly enough for a non-specialist reader to follow

## What Success Looks Like

Success in this repo is not only a strong score.

It looks like this:

- strong per-protocol performance
- clean separation of experiment families
- no silent dependence on weak preprocessing shortcuts
- stable run records
- explicit caveats around derived data and class imbalance
- documentation that explains both the code and the reasoning

## What This Repo Is Not Trying To Do By Default

Non-goals of the default benchmark path:

- default pooled image+video training
- treating extracted frames as if they were the original dataset
- mixing source test sets back into training
- reporting inflated results from weak split logic
- relying on undocumented legacy helpers as the main active pipeline

## Why This Problem Definition Matters For A Thesis Or Paper

A thesis-grade project must show that the research question was framed correctly before the models were trained.

This repo therefore defines the real question as:

"How can image-domain and video-domain deepfake detection be studied under defensible data handling, reproducible splits, and clearly separated spatial versus spatial+temporal experiment families?"

That is a stronger and more honest problem statement than simply saying:

"Build a deepfake detector."

## Reporting Guidance

When writing the thesis or paper, this stage should make the following points explicit:

- deepfake detection is not one homogeneous task
- this repo intentionally separates image, raw-video, and derived-frame workflows
- the project values methodological defensibility as much as model performance
- protocol separation is a design decision, not an accident
