# Research Knowledge Base

This directory is the working knowledge base for the deepfake detection research lifecycle.
It is intended to keep the project grounded in repo truth while also documenting reasoning,
decisions, risks, and experimental structure for future paper writing.

## Core Documents

- `research_notes.md`
- `experiment_matrix.md`

## Lifecycle Phases

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

## Ground Rules

- The codebase remains the source of truth for implementation details.
- `data/dataset.md` remains the canonical dataset and pipeline reference.
- `data/commands.md` remains the canonical command reference.
- This `books/` directory is for research framing, rationale, lifecycle tracking, and paper-grade notes.
- When implementation changes, update the relevant lifecycle phase note as well as the canonical technical docs.

## Current State

- New image training is active under `train/`.
- The active image smoke-test entrypoint is `python -m train.image_simulate`.
- The active implemented image trainer entrypoint is `python -m train.run_image_vit`.
- Video planning documents exist under `train/`, while migrated video execution code is not yet implemented there.
