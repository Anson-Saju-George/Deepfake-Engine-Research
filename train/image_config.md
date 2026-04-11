# Image Experiment Config

This file is the working experiment register for image-only training in the new `train/` tree.

Purpose:

- keep the planned image-model experiments in one place
- record why each model is being used
- record how each model is intended to be trained
- preserve failed experiments as part of the research trail instead of deleting them from history

## Global Image Training Policy

- protocol: `image_only`
- task type: image-domain spatial deepfake / synthetic-image detection
- current active image implementation:
  - smoke test: `python -m train.image_simulate`
  - ViT trainer: `python -m train.run_image_vit`
  - command sheet: `train/commands.md`
- split strategy:
  - preserve source dataset `train/test` boundaries
  - derive validation from source `train` when no explicit validation split exists
  - no pooled synthetic `70/10/20` split for image datasets
- default epochs: `10`
- default batch size: `64`
- balancing: enabled after correct protocol and split selection
- intended datasets:
  - `cifake`
  - `ai-generated-images-vs-real-images`
  - optional combined-image runs across both datasets

## Current Save Layout

- family-based save root:
  - `train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/`
- implemented family directory names:
  - `ViT`
  - `ConvNeXt`
  - `EfficientNet`
  - `ResNet`
- current dataset tags:
  - `cifake`
  - `ai_gen`
  - `image_combined`

Example:

- `train/image/ViT/IMG-EXP-01_vit_base_patch16_224_image_combined/`

## Checkpoint Policy

- keep `best.pth`
- keep `last.pth`
- save `config.json`
- save `history.csv`
- save `best_summary.json`
- save `final_summary.json`
- save `split_summary.json`
- save `run_record.md`
- create a promoted final checkpoint only when validation accuracy reaches at least `0.80`

## Current Pipeline Validation Status

- `python -m train.image_simulate` completed successfully on:
  - `cifake`
  - `ai-generated-images-vs-real-images`
- current image dataloader behavior is validated for:
  - raw image discovery
  - per-dataset `train/val/test` construction
  - one full train epoch plus validation and test passes
- oversized images in `ai-generated-images-vs-real-images` are now handled without PIL warning noise

## Dataset Use Notes

### `cifake`

- balanced large-scale synthetic-vs-real image benchmark
- useful for stable baseline training
- good for architecture comparison under clean class balance

### `ai-generated-images-vs-real-images`

- slightly imbalanced after corruption cleanup, but still close to balanced
- useful for testing generalization on a second image domain
- should remain on its original source `train/test` split

### Combined Image Setting

- combines `cifake` and `ai-generated-images-vs-real-images`
- preserves source dataset boundaries per dataset
- useful for testing whether a model scales better across image-generation styles
- should still be reported as an image-only experiment, not as multimodal training
- current dataset tag: `image_combined`

## Capacity Selection Logic

The image plan now follows a staged-capacity rule rather than selecting only the largest model immediately.

Working rule:

- start from the legacy lower bound already used in this repo
- if the model performs well but does not cross the desired threshold, escalate to the next higher-capacity model in the same family
- keep the failed or underperforming run in the experiment record

Chosen experiment ladders:

- ViT: `Base -> Large -> Huge`
- ConvNeXt: `Base -> Large -> XL`
- EfficientNet: `B6 -> B7`
- ResNet: `101 -> 152`

Reasoning:

- legacy image configs already used `ViT-Base`, `ConvNeXt-Base`, `EfficientNet-B4`, and `ResNet-50`
- the new plan intentionally starts at or above those prior floors
- `ViT-Tiny`, `ViT-Small`, `ConvNeXt-Tiny`, and `ConvNeXt-Small` are intentionally excluded from the main new plan
- lower EfficientNet variants and lower ResNet variants are intentionally excluded from the main new plan

## Experiment Register

### IMG-EXP-01

- experiment number: `IMG-EXP-01`
- model family: `ViT`
- model name: `vit_base_patch16_224`
- params: `~86M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- legacy image training already used ViT-Base
- useful for testing whether strong transformer capacity improves global patch-based reasoning on synthetic-image artifacts
- how to interpret it:
- if it performs strongly across both image datasets, it becomes a serious transformer baseline
- if it is promising but under target, escalate to `ViT-Large`
- status: `planned`

### IMG-EXP-02

- experiment number: `IMG-EXP-02`
- model family: `ViT`
- model name: `vit_large_patch16_224`
- params: `~307M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- second-stage transformer escalation above the legacy floor
- useful if ViT-Base is clearly capacity-limited but promising
- how to interpret it:
- if it materially improves over `IMG-EXP-01`, it becomes the main transformer candidate
- if it still underperforms, record that the transformer family was scaled and still did not justify larger cost
- status: `planned`

### IMG-EXP-03

- experiment number: `IMG-EXP-03`
- model family: `ViT`
- model name: `vit_huge_patch14_224`
- params: `~632M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- maximum planned transformer-scale experiment
- useful only if the ViT family continues to justify scaling
- how to interpret it:
- if gains are small, record diminishing returns for the ViT family
- status: `planned`

### IMG-EXP-04

- experiment number: `IMG-EXP-04`
- model family: `ConvNeXt`
- model name: `convnext_base`
- params: `~89M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- accepted legacy ConvNeXt floor
- strong modern convolutional baseline for local artifact and texture-focused cues
- how to interpret it:
- if it is strong but below target, escalate to `ConvNeXt-Large`
- status: `planned`

### IMG-EXP-05

- experiment number: `IMG-EXP-05`
- model family: `ConvNeXt`
- model name: `convnext_large`
- params: `~198M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- second-stage ConvNeXt escalation
- useful if the base variant is strong but still limited
- how to interpret it:
- if it is strong but below target, escalate to `ConvNeXt-XL`
- status: `planned`

### IMG-EXP-06

- experiment number: `IMG-EXP-06`
- model family: `ConvNeXt`
- model name: `convnext_xlarge`
- params: `~350M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- maximum planned ConvNeXt-scale experiment
- use only if the family still justifies scaling
- how to interpret it:
- if gains are small, record diminishing returns for the family
- status: `planned`

### IMG-EXP-07

- experiment number: `IMG-EXP-07`
- model family: `EfficientNet`
- model name: `tf_efficientnet_b6`
- params: `~43M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- start the EfficientNet ladder above the old B4 floor
- higher-capacity efficient-CNN baseline
- how to interpret it:
- if it is strong but below target, escalate to `EfficientNet-B7`
- status: `planned`

### IMG-EXP-08

- experiment number: `IMG-EXP-08`
- model family: `EfficientNet`
- model name: `tf_efficientnet_b7`
- params: `~66M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- maximum planned EfficientNet-scale experiment
- useful if the family still looks competitive after B6
- how to interpret it:
- if gains are small, document diminishing returns for the family
- status: `planned`

### IMG-EXP-09

- experiment number: `IMG-EXP-09`
- model family: `ResNet`
- model name: `resnet101.a1h_in1k`
- params: `~44.5M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- classical CNN baseline above the old ResNet-50 floor
- useful for paper comparability with a stronger-than-default ResNet baseline
- how to interpret it:
- if it is strong but below target, escalate to `ResNet-152`
- status: `planned`

### IMG-EXP-10

- experiment number: `IMG-EXP-10`
- model family: `ResNet`
- model name: `resnet152.a1_in1k`
- params: `~60.2M`
- epochs: `10`
- batch size: `64`
- protocol: `image_only`
- intended dataset plan:
- CIFAKE-only
- AI-generated-only
- combined image datasets
- why use it:
- maximum planned ResNet-scale experiment
- useful if the classical CNN family still justifies scaling
- how to interpret it:
- if gains are small, record diminishing returns for the family
- status: `planned`

## Escalation Reference

### ViT

- `ViT-Base (~86M)` -> start here
- `ViT-Large (~307M)` -> next escalation
- `ViT-Huge (~632M)` -> only if clearly justified later

### ConvNeXt

- `ConvNeXt-Base (~89M)` -> start here
- `ConvNeXt-Large (~198M)` -> next escalation
- `ConvNeXt-XL (~350M)` -> only if clearly justified later

### EfficientNet

- `EfficientNet-B6 (~43M)` -> start here
- `EfficientNet-B7 (~66M)` -> next escalation

### ResNet

- `ResNet-101 (~44.5M)` -> start here
- `ResNet-152 (~60.2M)` -> next escalation

## Logging Rules

For every image experiment, keep the entry even if the run fails.

Minimum fields to append later for each run:

- actual config file
- actual dataset scope
- seed
- optimizer and LR schedule
- best validation result
- final test result
- failure mode if unsuccessful
- notes on what changed from the initial plan

Current implemented record files per run directory:

- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`

## Current Intent

This register is mirrored in `train/image_model.py`. The Python registry and the implemented trainer should stay aligned with:

- experiment numbers
- family names
- dataset tags
- save layout
- checkpoint policy
