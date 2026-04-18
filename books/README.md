# Research Knowledge Base

This directory is the thesis-facing and journal-facing knowledge base for the repository.

Its job is to translate repository truth into research logic without drifting away from the actual codebase.

## What This Directory Must Do

Each file in `books/` should explain:

- what the concept is
- why it matters for deepfake detection
- what this repository currently does
- what methodological risk or tradeoff exists
- what the thesis or paper should explicitly report

That standard matters because this project is intended to be:

- dense enough for serious research reporting
- beginner-friendly enough for onboarding
- systematic enough for a thesis chapter structure
- concrete enough to stay anchored to code truth

## Reading Order

1. `01_problem_definition.md`
2. `02_data_collection.md`
3. `03_data_cleaning_preprocessing.md`
4. `04_exploratory_data_analysis.md`
5. `05_feature_engineering.md`
6. `06_model_selection.md`
7. `07_model_training.md`
8. `08_model_evaluation.md`
9. `09_hyperparameter_tuning.md`
10. `10_deployment.md`
11. `11_monitoring_maintenance.md`
12. `12_iteration_continuous_improvement.md`

Companion audit documents:

- `research_notes.md`
- `experiment_matrix.md`

## Current Repo Truth

At the current project state:

- the active dataloader is `data/dataloader.py`
- the active image training tree is `train/image/`
- the active image registry is `train/image/image_models.py`
- the active image command sheet is `train/image/image_commands.md`
- the active video research registry is `train/video/video_models.py`
- the active video command sheet is `train/video/video_commands.md`
- the active raw-video smoke path is `train/video/simulate_video_train.py`
- raw-video training remains the intended main video path
- frame materialization remains auxiliary rather than mandatory
- `train_old/` remains reference and historical context, especially for older video behavior

## Current Headline Findings

The present documentation should be read with these validated results in mind:

- best completed image result:
  - `IMG-EXP-04` ConvNeXt-Base
  - test F1: `0.9863`
  - test accuracy: `0.9863`
- best completed video result:
  - `VID-TMP-02` ConvNeXt-Large sequence
  - test F1: `0.7841`
  - test accuracy: `0.9089`

Important current interpretation:

- image detection is presently much easier than raw-video detection in this repository
- CNN-style backbones currently outperform the completed transformer-family runs in both image and video results
- the active `temporal` and `spatiotemporal` video branches currently converge to the same clip-based aggregation mechanics
- reserved native-video IDs `VID-ST-07..12` remain future work rather than active evidence-bearing experiments

## Relationship To Other Docs

Canonical technical reference:

- `data/dataset.md`
- `data/commands.md`

Training implementation and command reference:

- `train/image/image_model_config.md`
- `train/image/image_commands.md`
- `train/video/video_config.md`
- `train/video/video_commands.md`
- `train/video/Experiment_List.md`

This `books/` directory serves a different role:

- it explains why design decisions matter
- it records risks, caveats, and reporting guidance
- it converts repo behavior into thesis-ready narrative
- it helps prevent code and documentation from drifting apart

## Practical Use

For onboarding:

- read from `01_problem_definition.md` forward

For experiment audit:

- pair `research_notes.md` with `experiment_matrix.md`

For thesis or paper writing:

- use each lifecycle file as the scaffold for a chapter or section
- cross-check every claim against `data/` and `train/` before final writing
