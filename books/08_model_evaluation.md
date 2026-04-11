# Model Evaluation

## Evaluation Principles

- keep evaluation aligned with the protocol used during training
- do not compare image and video experiments as if they were the same task
- report dataset-specific and protocol-specific results clearly
- report video spatial baselines separately from spatial+temporal sequence models

## Expected Reporting Areas

- accuracy
- precision
- recall
- F1
- ROC-AUC when appropriate
- confusion matrices
- per-dataset breakdowns
- run-directory records containing config, split summary, and checkpoint references

## Current New-Tree Image Record Format

Each completed image run should retain:

- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`
- `best.pth`
- `last.pth`

## Evaluation Caveats

- frame-derived experiments should be labeled as derived-video experiments
- results on fake-heavy video corpora should be interpreted alongside class balance and balancing strategy
- split methodology must be stated explicitly in the paper
- image results should state that source train/test boundaries were preserved and validation was derived from training data where needed
- video and frame-derived results should state that identity-aware `70/10/20` splitting was used
