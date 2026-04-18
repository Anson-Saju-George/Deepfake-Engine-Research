# Monitoring & Maintenance

## What This Stage Means

Monitoring and maintenance are the disciplines that keep a research codebase truthful over time.

For someone new to this field, it is easy to assume maintenance only means fixing bugs after code breaks.
In a machine learning research repository, maintenance is broader than that.
It includes the continuous work required to keep the following aligned:

- the actual files on disk
- the data discovered by the loader
- the split logic used by experiments
- the behavior of the training scripts
- the claims made in the documentation
- the claims that will later appear in a thesis or paper

If those pieces drift apart, the repository may still run, but the scientific story stops being trustworthy.

## Why Maintenance Is Especially Important In This Project

This project has several characteristics that make silent drift a real risk.

It contains:

- multiple data modalities
- multiple dataset families
- optional derived data such as frame folders
- active preprocessing utilities
- legacy and new training trees coexisting during migration
- a documentation layer intended to support thesis-grade reporting

Each of those increases flexibility, but each also increases the chance of mismatch.

Examples of the kinds of drift that can happen:

- dataset counts change after cleaning, but the docs still show older numbers
- a dataloader starts discovering derived frames, but a paper table still describes only raw data
- a trainer changes save layout, but run-record documentation still points to the old path
- a protocol default changes, but the commands documentation still teaches the previous behavior

This is why maintenance in this repository is not optional housekeeping. It is part of research integrity.

## What Monitoring Means Here

Monitoring means repeatedly asking whether the current repository state still matches the intended methodology.

That includes questions such as:

- do filesystem counts still match what the dataloader sees?
- are raw dataset totals still what the docs claim they are?
- are there newly corrupted files?
- are frame extraction artifacts being created in ways that change loader discovery?
- do training outputs still match the documented save structure?
- do current run records still match what the trainer actually writes?

These are not merely engineering checks. They are evidence checks.

## The Main Kinds Of Drift This Repo Must Guard Against

### Dataset Drift

Dataset drift in this repository means the effective corpus changes but the surrounding documentation and experiment assumptions do not.

Examples:

- deleting corrupted files
- adding newly extracted frame folders
- moving files into temp or archive locations
- introducing partial derived outputs

Why this matters:

- published totals become wrong
- class distributions shift
- comparisons between old and new runs become harder to interpret

### Loader Drift

Loader drift means the data layer starts discovering or interpreting data differently from what the team thinks it is doing.

Examples:

- a change in protocol defaults
- a new discovered media type
- an auxiliary path becoming visible to the main analysis unintentionally

Why this matters:

- training can silently change without an obvious code failure
- the repo can still run while the experimental meaning has changed

### Documentation Drift

Documentation drift means the Markdown layer stops matching the code and actual dataset state.

Examples:

- docs describing full-frame materialization as core when the repo has moved to raw-video training as primary
- docs omitting an implemented model family

Why this matters:

- future readers lose trust
- paper writing becomes harder
- collaborators cannot tell which statements are current

### Run Artifact Drift

Run artifact drift means output folders, summaries, configs, and checkpoints no longer agree with each other.

Examples:

- `config.json` points to an old save path
- run record says a different dataset scope than the actual folder naming
- best checkpoint metadata does not match the summary file

Why this matters:

- reproducibility becomes weaker
- later deployment or comparison becomes error-prone

## The Main Monitoring Tools In This Repo

This repository already includes audit scripts that should be treated as part of the scientific control surface.

### `python -m data.dataset_analyzer`

What it does:

- inspects configured datasets on disk
- compares raw on-disk counts with loader discovery
- reports label counts and dtype counts

Why it matters:

- it is the main check that raw dataset truth and loader discovery still agree
- it catches the difference between total discoverable samples and the intended raw corpus

### `python -m data.video_frame_stats`

What it does:

- opens raw videos
- inspects frame counts and durations
- reports dataset-level frame mass and label balance by frame mass

Why it matters:

- it helps interpret video corpora beyond simple video counts
- it makes clear whether class imbalance remains strong even when measured in frame mass
- it prevents unsupported assumptions about “balanced enough” video data

### `python -m data.image_video_frame`

What it does:

- audits image, video, and frame views together when needed

Why it matters:

- it helps verify multi-view corpus behavior
- it is useful when derived frame artifacts exist and need to be interpreted carefully

### `python train_data_pipeline_pull.py`

What it does:

- reports protocol-aware split and distribution information through the active data pipeline

Why it matters:

- it connects raw dataset state to training-time behavior
- it shows what the trainer will actually see, not just what exists on disk

## What A Healthy Maintained State Looks Like

A healthy state in this repository means all of the following are true at once.

- raw configured datasets still match between disk and dataloader discovery
- cleaned dataset totals are still the documented totals for the raw corpus
- optional derived artifacts are recognized as auxiliary rather than silently treated as raw corpus truth
- split policy in docs matches split policy in code
- protocol defaults in docs match protocol defaults in code
- run directories contain the files the trainer claims to write
- run summary files agree with actual checkpoint identity and folder location

A repository can fail one of these checks while still “working.”
That is exactly why maintenance has to be explicit.

## Maintenance Rules This Project Should Continue To Follow

Whenever one of the following changes, documentation should be updated in the same work cycle:

- dataset counts
- dataset cleaning history
- split strategy
- protocol defaults
- active preprocessing path
- training save layout
- implemented experiment families
- accepted or rejected methodological branches

This rule matters because delayed documentation nearly always becomes stale documentation.

## Why Maintenance Is Part Of Research Integrity

A thesis, dissertation, or journal paper is not only judged by whether the model score is high.
It is also judged by whether the reported method is traceable and internally consistent.

Maintenance contributes to that because it preserves:

- reproducibility
- provenance
- methodological honesty
- interpretability of later comparisons

If the repository drifts while the written methodology remains frozen, the paper may stop describing the code that actually produced the results.
That is not a minor inconvenience. That is a scientific reliability problem.

## What The Final Thesis Or Paper Should Be Able To Claim

The final write-up should be able to state, credibly, that:

- the dataset state was repeatedly audited
- cleaned counts were verified against loader discovery
- derived artifacts were treated explicitly, not silently
- run records were preserved in a structured way
- documentation was updated as methodology evolved

Those claims make the work stronger because they show that the project was managed as a controlled research process rather than a loose collection of scripts.

## Current Repo Truth

At the present stage:

- the canonical raw corpus remains `192,016` samples
- the main image training path is active in the new `train/` tree
- the video methodology is documented clearly even though its new execution layer is still transitioning
- the documentation layer itself has become part of the controlled research artifact

That means monitoring and maintenance are no longer side tasks in this repo. They are part of the methodology.
