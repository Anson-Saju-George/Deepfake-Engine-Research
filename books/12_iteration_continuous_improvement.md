# Iteration / Continuous Improvement

## What This Stage Means

Iteration is the process of improving a research system without destroying the ability to understand why it improved.

For beginners, iteration in machine learning often looks like this:

- try another model
- change the data
- increase epochs
- add augmentation
- change optimizer
- rerun and hope the score rises

That approach can generate movement, but it does not reliably generate knowledge.

In a serious research workflow, iteration should answer a clearer question:

- what changed?
- why was it changed?
- what stayed fixed?
- what did the result teach us?

That is the standard this repository is trying to follow.

## Why Iteration Must Be Controlled

A result only becomes useful to a thesis or paper if it is interpretable.

If architecture, preprocessing, data scope, split logic, and optimizer all change at the same time, then even a stronger result is hard to explain.
Was the gain caused by:

- the model family?
- the data scope?
- the split?
- the regularization?
- the augmentation?
- randomness?

Without control, the answer is usually unclear.

That is why this project prefers staged iteration rather than uncontrolled exploration.

## The Iteration Order Used In This Repo

The intended order of progress in this repository is disciplined.

1. verify raw dataset truth
2. verify cleaning state
3. verify preprocessing semantics
4. verify split correctness
5. establish baseline models
6. evaluate baseline behavior
7. tune selectively
8. expand to ablations only after baseline stability

This order matters because later changes only mean something if earlier assumptions are already reliable.

For example:

- tuning a model before split correctness is established is low-value
- deploying a model before protocol semantics are stable is premature
- comparing frame-derived runs to raw-video runs before clearly labeling them is misleading

## What Counts As A Good Iteration In This Project

A good iteration changes one meaningful factor while keeping the rest interpretable.

Examples:

- changing the backbone family while keeping the optimizer stack fixed
- changing dataset scope while keeping split logic fixed
- changing sequence length while keeping the temporal sampling rule fixed
- testing balancing on and off after the protocol boundaries are already correct

Why this is good:

- it allows cause-and-effect reasoning
- it produces evidence that can be explained later in a paper
- it reduces the chance of accidental benchmark inflation through confounded changes

## What Counts As A Bad Iteration In This Project

A bad iteration mixes too many moving parts or changes important assumptions silently.

Examples:

- changing architecture, optimizer, and augmentation all at once
- changing dataset scope and split strategy together without documentation
- changing preprocessing behavior without updating the docs
- reporting a new top-line score without recording what changed

Why this is bad:

- it makes gains hard to attribute
- it weakens the scientific narrative
- it produces confusion instead of cumulative knowledge

## The Improvement Philosophy Behind The New Image Tree

The new image training tree reflects a deliberate iteration strategy.

The project is currently:

- using one shared optimizer and training-control stack across image families
- comparing different backbones under that same recipe
- keeping run records structured and explicit
- preserving failed or weaker runs as part of the evidence trail

This is important because it means the first phase of iteration is not “search for the best possible trick.”
It is “establish a clean comparative baseline.”

That is the right first move for a thesis-grade benchmark pipeline.

## Why Rejected Paths Must Also Be Recorded

A mature research workflow should document not only what worked, but also what was rejected and why.

This repository already has examples of that thinking.

Examples include:

- treating full frame materialization as auxiliary rather than mandatory for main video training
- evaluating GPU-accelerated extraction paths and not adopting them as the main route when they were not reliable enough
- refusing to treat pooled image+video training as the default benchmark path

Why this matters:

- it prevents the project from repeating already-understood dead ends
- it gives the thesis a more honest methodological story
- it shows that decisions were reasoned, not arbitrary

## What Continuous Improvement Should Focus On Next

Given the present repository state, the most useful continuous-improvement directions are not random.
They are relatively clear.

### Near-Term High-Value Iteration Areas

- continue image-family benchmarking under the current controlled recipe
- finish documenting completed runs to a publication-quality standard
- preserve raw-video-first methodology for video experiments
- add video benchmarks in a way that mirrors the clarity already achieved on the image side
- expand the completed runner surface carefully beyond the current validated image-style video backbones

### Later Improvement Areas

- family-specific tuning after baseline comparison
- stronger video ablations beyond the current clip-based implementation
- threshold and calibration analysis for inference-oriented reporting
- possible deployment packaging after the benchmark story is stable

## What Iteration Already Taught In This Repo

The completed work already gives a strong iteration story.

What worked:

- starting with image models first gave a clean benchmark and a stable optimization recipe
- keeping one shared image recipe exposed architecture differences cleanly
- moving to raw-video training exposed the real difficulty gap between image and video detection
- lowering sequence length and base learning rate stabilized the strongest large clip-based ConvNeXt runs
- train-only oversampling remained the strongest default imbalance strategy in the completed comparisons

What did not work as hoped:

- FFmpeg QSV did not become the stable fast default on this corpus
- Decord remained unsafe as a default because damaged H.264 videos could crash workers
- focal loss did not improve the completed spatial ConvNeXt comparison
- early hybrid/spatiotemporal claims were too broad relative to the actual current implementation

What this means:

- the project improved not by adding complexity everywhere
- it improved by removing weak assumptions and keeping only the choices that survived actual runs

## Why Continuous Improvement Is A Documentation Problem Too

Iteration is not only a modeling process. It is also a documentation process.

Every serious iteration should leave behind a readable record of:

- what changed
- why it changed
- what was held constant
- what the outcome was
- what decision followed from that outcome

If that chain is missing, the repository may still contain code and checkpoints, but it will be much harder for a future reader to reconstruct the reasoning behind the work.

That is exactly the kind of problem this knowledge base is meant to prevent.

## What The Thesis Or Paper Should Show About Iteration

The final thesis or paper should communicate that the project improved through staged control rather than trial-and-error.

That means the written story should make clear that:

- the data layer was verified before major training claims were made
- preprocessing decisions were audited before being normalized into the pipeline
- baseline model families were compared under one shared recipe before aggressive tuning began
- rejected paths were documented rather than quietly forgotten
- later improvements built on earlier validated choices

That makes the final research narrative much stronger because it shows not only what result was achieved, but how methodological confidence was built over time.

## Current Repo Truth

At the present stage:

- image training in the new `train/` tree is active
- multiple image backbone families are implemented under one common recipe
- video training in the new `train/` tree is active for the current image-style video backbones
- the current strongest validated video story is clip-based ConvNeXt rather than a broad three-paradigm claim
- the documentation layer itself is now part of the continuous-improvement process

That means iteration in this repository is not just about pushing scores higher. It is about making every next step more defensible than the previous one.
