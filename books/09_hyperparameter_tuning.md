# Hyperparameter Tuning

## Purpose

Tune models only after the split logic, preprocessing path, and evaluation methodology are stable.

## Likely Tuning Axes

- learning rate
- batch size
- epochs
- seq_len
- augmentation strength
- balancing strategy
- optimizer and scheduler settings

Current fixed image baseline defaults in the new tree:

- epochs: `10`
- batch size: `64`
- optimizer: `AdamW`
- scheduler: cosine annealing
- label smoothing: `0.1`
- EMA: enabled

## Guardrails

- do not tune on test performance
- keep search space documented
- preserve seed and config traceability

## Documentation Expectation

Every meaningful tuning run should record:

- config key
- changed hyperparameters
- datasets
- protocol
- best validation result
- final test result
