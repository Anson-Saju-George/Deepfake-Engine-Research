# Model Evaluation

## What This Stage Means

Model evaluation is the stage where the project stops asking, “can the model train?” and starts asking, “what exactly has the model learned, and how trustworthy is that result?”

For someone new to the field, evaluation is not just one final accuracy number.
It is the structured process of checking whether a trained model performs well on held-out data and whether that apparent performance is meaningful.

In serious research, evaluation answers questions such as:

- how well does the model generalize beyond training data?
- is the model balanced across classes or skewed toward one side?
- is the result stable across datasets or only strong on one subset?
- are we evaluating the same protocol we trained?
- does the selected checkpoint reflect a defensible criterion?

That is why evaluation must be treated as a methodological chapter, not as an afterthought.

## Why Evaluation Must Be More Than Accuracy

Accuracy is useful, but it is not a complete description of model behavior.

A model can appear strong by accuracy while still being problematic if:

- it favors one class too heavily
- it has poor recall on manipulated samples
- it produces overly confident but brittle decisions
- it performs well on one dataset and weakly on another inside a combined run

For that reason, this repository documents a broader evaluation view. The goal is not just to produce a high number. The goal is to produce an interpretable and defensible result.

## The First Evaluation Principle In This Repo: Protocol Alignment

Evaluation must match the protocol used during training.

If a model is trained under `image_only`, then evaluation should be described as an image-domain evaluation.
If a model is trained under `video_only`, then evaluation should be described as a video-domain evaluation.
If a model is trained under `frame_only`, then evaluation should be described as a derived-frame evaluation.

Why this matters:

- image, raw-video, and derived-frame tasks are not the same scientific task
- a score from one protocol should not be casually compared as if it came from another
- collapsing modality-specific results into one loose benchmark would make the paper weaker

## The Second Evaluation Principle: Media Mode Matters

Within video-related work, the evaluation mode must also be stated clearly.

### `single` Mode

What it means:

- evaluation is based on single sampled frames from raw videos

What kind of model behavior it measures:

- spatial cue exploitation in a video-domain setting

Why it should be reported separately:

- this is not the same as temporal reasoning
- it is a valid and useful baseline, but it should not be mislabeled as a fully temporal video result

### `sequence` Mode

What it means:

- evaluation is based on contiguous ordered clips sampled from raw videos

What kind of model behavior it measures:

- spatial plus temporal reasoning

Why it should be reported separately:

- it answers a different question from single-frame video evaluation
- it is methodologically stronger for temporal artifact analysis, but also more complex

## What Metrics This Repo Expects To Report

The repository expects evaluation to consider more than one metric.

The main reporting areas are:

- accuracy
- precision
- recall
- F1
- ROC-AUC when appropriate
- confusion matrix analysis
- per-dataset breakdowns when combined scopes are involved
- structured run records tying metrics back to checkpoints and splits

## What Each Main Metric Means And Why It Matters

### Accuracy

What it is:

- the proportion of total predictions that were correct

Why it is useful:

- easy to understand
- easy to compare across runs
- useful as a headline summary

Why it is insufficient on its own:

- it can look good even when one class is favored too much
- it does not directly reveal the precision/recall tradeoff

### Precision

What it is:

- among predicted positives, how many were actually positive

Why it matters in deepfake detection:

- a model with low precision may flag too many real samples as fake
- that makes the system operationally noisy and less trustworthy

### Recall

What it is:

- among actual positives, how many were recovered by the model

Why it matters in deepfake detection:

- a model with weak recall may miss too many fake samples
- such a model can look conservative or “safe,” but still fail the central detection task

### F1

What it is:

- the harmonic mean of precision and recall

Why it matters here:

- it captures a more balanced operational view than accuracy alone
- it becomes especially useful when class behavior is not perfectly symmetric
- it resists being overly flattering when a model is strong on one side but weak on the other

This is exactly why the current image training path selects best checkpoints by `val_f1`.

### ROC-AUC

What it is:

- a threshold-independent ranking metric that reflects separability across decision thresholds

Why it can help:

- useful when later threshold tuning or calibration becomes important
- useful for understanding whether a model’s ranking ability is stronger than a single chosen operating threshold suggests

### Confusion Matrix

What it is:

- a class-by-class count of prediction outcomes

Why it matters:

- it makes the failure structure visible
- it shows whether false positives or false negatives dominate
- it helps explain why a model with decent accuracy may still be misbehaving

### Per-Dataset Breakdown

What it is:

- evaluating constituent datasets separately inside broader scopes such as combined image or combined video runs

Why it matters:

- a combined score can hide weakness on one dataset
- different datasets encode different biases and artifact structures
- a paper that reports only pooled performance may overstate generality

## Why Best Checkpoint Selection Uses Validation F1

The current new-tree image policy chooses the best checkpoint by validation F1.

Why not just use validation accuracy?

- accuracy is simpler, but less sensitive to imbalanced decision behavior
- a model can achieve strong accuracy while still producing a weak precision/recall balance
- F1 gives a more defensible balance-aware selection rule for binary classification

This does not mean accuracy is unimportant. It means accuracy is not the only thing that should decide which checkpoint is considered best.

## Why The Promoted Final Checkpoint Uses A Validation Accuracy Threshold

The new image trainer also uses a practical artifact-management rule:

- a promoted final checkpoint is only exported when validation accuracy is at least `0.80`

Why this exists:

- it helps manage artifacts
- it creates a practical boundary for what counts as obviously usable or promotable
- it keeps the filesystem cleaner than exporting every run as if it were equally strong

Why this is not the main scientific selection rule:

- best checkpoint identity is still governed by validation F1
- the threshold is a practical packaging gate, not the primary research criterion

## Why Split Policy Must Always Be Mentioned In Evaluation

No evaluation result is complete without split context.

This repo requires the reported result to make clear:

- what split policy was used
- whether the task was image, raw-video, or derived-frame based
- whether balancing was enabled
- whether balancing was train-only or applied more broadly
- whether the evaluation was `single` or `sequence` when relevant

Why this matters:

- a score is only meaningful relative to the split protocol behind it
- image source-boundary preservation and video identity-aware splitting are scientifically different procedures
- omitting split context makes comparisons weaker and easier to misread

## The Required Evaluation Statements For A Thesis Or Paper

Every serious results section based on this repository should state the following for each table or figure.

- protocol used: `image_only`, `video_only`, `frame_only`, or `combined_aux`
- media mode used: `single` or `sequence`, where relevant
- split policy used
- whether balancing was enabled
- whether balancing was restricted to the training split
- which metric selected the checkpoint
- whether the result is image-based, raw-video-based, or derived-frame-based
- whether the result reflects a single dataset or a combined dataset scope

## Active Video Evaluation Methodology In This Repo

For the active `video_only` path, the defensible evaluation rule is:

- split videos by identity first
- oversample only the training split if balancing is enabled
- leave validation untouched
- leave test untouched

Why this is the correct policy:

- the training loader may need rebalance so the model does not collapse toward the majority class
- validation and test must remain natural so the reported result reflects realistic class behavior
- oversampling evaluation data would inflate or distort the apparent operating behavior of the model

This means a video run can legitimately use train balancing while still being evaluated on a naturally skewed held-out set.

## Evaluation Risks This Repo Explicitly Tries To Avoid

### Mistake 1: Treating Image And Video Scores As Directly Interchangeable

Why it is wrong:

- image and video are not the same task
- the data-generating process, split logic, and temporal structure differ

### Mistake 2: Treating Frame-Derived Results As Native Raw-Video Results

Why it is wrong:

- derived frames come from a preprocessing pipeline with its own assumptions
- they are useful, but not methodologically identical to raw-video inference or training

### Mistake 3: Reporting Only Combined Scores

Why it is wrong:

- pooled scores can hide dataset-specific weakness
- a paper may sound stronger than the actual evidence if individual datasets are not also examined

### Mistake 4: Reporting Only Accuracy

Why it is wrong:

- it hides class-balance failures and skewed decision behavior
- it makes checkpoint selection rationale look weaker

## What A Good Evaluation Record Looks Like In This Repo

The new image training path already writes structured evidence for completed runs.

Each completed image run should retain:

- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`
- `best.pth`
- `last.pth`

Why this matters for evaluation:

- metrics are linked back to the actual run configuration
- split counts are preserved, not guessed later
- best-checkpoint logic is documented
- the run can be audited later without reverse-engineering the entire folder manually

## How Evaluation Should Be Interpreted For Combined Datasets

Combined dataset evaluation should always be treated carefully.

Why:

- combined image datasets may contain different difficulty profiles and generation characteristics
- combined video datasets may contain different temporal and class-balance properties
- one strong constituent dataset can elevate a pooled metric and hide weakness elsewhere

So a responsible reporting strategy is:

- report the combined result
- also report per-dataset or at least per-scope breakdowns wherever possible

## How Evaluation Should Be Interpreted For Fake-Heavy Video Corpora

The repository’s video analysis already shows that some video corpora are fake-heavy not only by video count, but also by frame mass.

Why this matters for evaluation:

- a strong result on a fake-heavy dataset may still require careful interpretation
- balancing strategy should be stated clearly
- class behavior should be discussed with more than one metric

## What The Thesis Or Paper Should Be Able To Claim About Evaluation

A strong final write-up should be able to say that:

- evaluation was protocol-aligned
- split policy was stated explicitly
- checkpoint selection used a balance-aware validation metric
- combined results were not used to hide dataset-specific weaknesses
- frame-derived results were labeled honestly
- run records preserved the evidence behind final reported numbers

## Current Repo Truth

At the present stage:

- the new image training path evaluates and records runs in a structured way
- best image checkpoints are selected by validation F1
- final image records preserve config, split summary, history, checkpoint references, and human-readable run notes
- the new video runner surface also evaluates and records completed runs in a structured way for the active image-style video backbones

## Current Saved Results Snapshot

Image and video results must be interpreted separately.

### Image

Current completed image results:

- `IMG-EXP-04` ConvNeXt-Base
  - best val F1: `0.9869`
  - test F1: `0.9863`
  - test accuracy: `0.9863`
- `IMG-EXP-07` Swin-Base
  - best val F1: `0.9861`
  - test F1: `0.9842`
  - test accuracy: `0.9842`
- `IMG-EXP-05` ConvNeXt-Large
  - best val F1: `0.9864`
  - test F1: `0.9840`
  - test accuracy: `0.9840`
- `IMG-EXP-01` ViT-Base
  - best val F1: `0.9734`
  - test F1: `0.9702`
  - test accuracy: `0.9703`
- `IMG-EXP-02` ViT-Large
  - best val F1: `0.9592`
  - test F1: `0.9546`
  - test accuracy: `0.9548`

Current image conclusion:

- `ConvNeXt-Base` is the strongest completed saved image result
- `Swin-Base` is very close
- `ConvNeXt-Large` did not improve over `ConvNeXt-Base`
- the saved ViT runs remain clearly below the strongest CNN and hierarchical-transformer image baselines
- the image-domain benchmark is close to saturation relative to the current datasets, because the strongest completed runs are already near `0.98` test F1

### Video

Current completed video results:

- spatial
  - `VID-SPA-02` ConvNeXt-Base, `loss=none`
    - best val F1: `0.7246`
    - test F1: `0.7023`
    - test accuracy: `0.8566`
- temporal
  - `VID-TMP-01` ConvNeXt-Base, `loss=none`
    - best val F1: `0.6998`
    - test F1: `0.7090`
    - test accuracy: `0.8652`
  - `VID-TMP-02` ConvNeXt-Large, `loss=none`, `seq_len=4`, `lr=5e-5`
    - best val F1: `0.7752`
    - test F1: `0.7841`
    - test accuracy: `0.9089`
- spatiotemporal
  - `VID-ST-02` ConvNeXt-Base hybrid
    - best val F1: `0.5977`
    - test F1: `0.5744`
    - test accuracy: `0.7663`
  - `VID-ST-03` ConvNeXt-Large hybrid, `seq_len=4`, `lr=5e-5`
    - best val F1: `0.7752`
    - test F1: `0.7841`
    - test accuracy: `0.9089`
  - `VID-ST-05` MaxViT-Base hybrid, `seq_len=4`, `lr=5e-5`
    - best val F1: `0.6848`
    - test F1: `0.6414`
    - test accuracy: `0.8186`

Current video conclusion:

- the strongest completed saved video result is `VID-TMP-02` ConvNeXt-Large temporal
- `VID-ST-03` matches it numerically under the current implementation
- `VID-ST-02` underperformed strongly
- `VID-ST-05` improved over `VID-ST-02` but still trails the ConvNeXt-Large sequence result

## Cross-Domain Interpretation

The most important current evaluation gap is not within one family. It is between image and raw-video detection.

Best completed saved results:

- image leader: `IMG-EXP-04` ConvNeXt-Base
  - test F1: `0.9863`
- video leader: `VID-TMP-02` ConvNeXt-Large sequence
  - test F1: `0.7841`

This gap is large enough to be a core research finding.

The current defensible interpretation is:

- image-domain detection in this repository is much easier than raw-video detection
- the image task likely contains stronger static artifact signal and/or stronger dataset bias
- raw-video detection is harder because weak frames, motion, compression, and temporal inconsistency dilute the most discriminative single-frame cues
- therefore a weaker video score should not be framed as simple model failure; it is evidence that the raw-video task is materially harder

Important implementation interpretation:

- the active `temporal` and `spatiotemporal` categories currently share the same sequence-trainer mechanics
- both use per-frame image-backbone encoding followed by temporal mean pooling
- therefore the current `ST` branch should not be overstated as a fundamentally different native-video model family

Practical reporting implication:

- the current implementation effectively yields two validated modeling paradigms:
  - spatial
  - clip-based
- the registry still keeps `temporal` and `spatiotemporal` IDs separate for experiment accounting
- but the thesis should explain clearly that the present completed `TMP` and `ST` evidence converges to one clip-based aggregation framework

That is the evaluation truth the documentation should preserve right now.
