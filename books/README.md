# Research Knowledge Base

This directory is the thesis-facing and journal-facing knowledge base for the repository.

Its job is to translate repository truth into research logic without drifting away from the actual codebase.

---

> ## ⚠️ Direction notice — read before using these chapters for the paper
>
> **The authoritative research design is now `papers/writing/EXPERIMENT_DESIGN.md`** (bidirectional
> cross-regime transfer: face-manipulation vs fully-synthetic video, whole-frame, audited).
> It **supersedes** the older image-vs-video modality framing that most chapters below were
> written under.
>
> When writing the new paper:
> - **Direction / claims** → `papers/writing/EXPERIMENT_DESIGN.md` + `papers/writing/READING_LIST.md` (authoritative).
> - **Project scope / layout** → root `PROJECT_CONTEXT.md`.
> - **All performance numbers** (env, throughput, model complexity) → `perf/PERFORMANCE_LOG.md`, with context in `perf/PERFORMANCE_CONTEXT.md`.
> - **The 22 completed runs below are the OLD framing** — archived evidence, at most a face-centric baseline. Do not reproduce the old thesis as the paper's contribution.
>
> These lifecycle chapters remain useful as method/background scaffolding and code-truth
> reference, but every claim must be reconciled against `EXPERIMENT_DESIGN.md` first.

---

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

Companion audit document:

- `research_notes.md`

(The former `experiment_matrix.md` was folded into `08_model_evaluation.md` → "Archived Experiment Surface".)

## Current Repo Truth

At the current project state:

- the active dataloader is `data/dataloader.py`
- the active image training tree is `train/image/`
- the active image registry is `train/image/image_models.py`
- the **video trainer tree has been archived** to `temp/legacy_models/train_video/` (the new
  design cuts the temporal-head sweep; `train/evaluate_all.py` still handles image runs and
  treats video evaluation as optional)
- under the new design, detection is **whole-frame** — frame materialization / face-cropping
  from the old pipeline is legacy, not the intended path

## Headline Findings (OLD framing — archived evidence)

These are the completed-run results under the superseded image-vs-video framing. Full
tables live in `books/08_model_evaluation.md`; all performance/complexity numbers in
`perf/PERFORMANCE_LOG.md`. Treat as a baseline, not the new paper's contribution:

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
- `VID-ST-07..09` (ConvLSTM/Hybrid Transformer/Hybrid TCN) are active evidence-bearing experiments with completed runs; only `VID-ST-10..12` remain future work

## Relationship To Other Docs

Canonical technical reference:

- `data/dataset.md`
- `data/commands.md`

Training implementation and command reference:

- `train/image/image_model_config.md`
- `train/image/image_commands.md`
- `train/video/video_config.md`
- `train/video/video_commands.md`
- `train/video/video_commands.md`

This `books/` directory serves a different role:

- it explains why design decisions matter
- it records risks, caveats, and reporting guidance
- it converts repo behavior into thesis-ready narrative
- it helps prevent code and documentation from drifting apart

## Practical Use

For onboarding:

- read from `01_problem_definition.md` forward

For experiment audit:

- pair `research_notes.md` with the "Archived Experiment Surface" appendix in `08_model_evaluation.md`

For thesis or paper writing:

- use each lifecycle file as the scaffold for a chapter or section
- cross-check every claim against `data/` and `train/` before final writing
