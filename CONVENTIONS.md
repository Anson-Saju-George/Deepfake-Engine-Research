# Conventions

Single source of truth for label convention, metric definitions, split policy, and clip-sampling
semantics. Written to stop the exact kind of silent mismatch this repo already had once (see
"Label convention" below) from happening again. If code and this document disagree, that's a bug —
in one of the two.

## Label convention

**Raw label values (never ambiguous):** `real = 1`, `fake = 0`, set by `data/dataloader.py:DATASET_CONFIG`
for every dataset, image and video alike. This part is not in question and does not change.

**What's genuinely unresolved: which class is "positive" for F1/precision/recall.** Unlike accuracy and
AUC (symmetric — same value regardless of which class you call positive), F1/precision/recall are NOT
symmetric, and this repo currently has two different answers to "which class is positive" living in it
at once:

- `train/eval_predictions_common.py` (the ACTIVE code computing metrics for every completed run today):
  `pos_label=1` → **real is positive**.
- `docs/code/metrics.py` (an unintegrated reference module): documents `"label 1 == FAKE, label 0 == REAL"`
  → **fake is positive** — the opposite — and reports F1 under both conventions explicitly rather than
  picking one, specifically to avoid this exact ambiguity.

**Status: deliberately deferred, by explicit user decision, until all refactor tasks (T0–T5) are complete.**
Not a technical blocker — a decision the user chose to hold. See the governing `FIXTURE_PLAN.md`, Open
Decision #1, which carries a standing reminder to raise this again once T0–T5 finish. Until it's resolved:

- Do not promote `docs/code/metrics.py` into the active trainers.
- Do not change `train/eval_predictions_common.py`'s `pos_label` convention.
- Any new F1/precision/recall number should be read alongside this section, not assumed to match a
  particular convention.

## Corpus composition & naming (FF++-derived training set)

**This corpus is NOT "FaceForensics++."** FF++ is defined, in every paper that cites it, as
the four-manipulation set (Deepfakes / Face2Face / FaceSwap / NeuralTextures). What's actually
on disk here (`datasets/videos/faceforensics++/`, confirmed via `test/results/corpus_profile.json`,
all 5,429 files):

| Subset | Videos | What it actually is |
|---|---:|---|
| `fake/Deepfakes` | 1,000 | The DF subset of FaceForensics++ proper |
| `fake/DeepFakeDetection` | 3,066 | The **Google/Jigsaw DeepFakeDetection (DFD) dataset** — a separate release, often distributed alongside FF++ but not one of its four core manipulations |
| `real/youtube` + `real/actors` | 1,363 | Real videos (FF++'s own youtube-source real videos + DFD's actor-source real videos) |

`Face2Face`/`FaceSwap`/`NeuralTextures` are not present in this corpus at all. **Call this
training set "a subset of FaceForensics++ (Deepfakes) combined with the DeepFakeDetection
dataset," cite both sources, and do not write "trained on FaceForensics++" anywhere it implies
the full four-manipulation benchmark** — a reviewer opening the paper expecting DF/F2F/FS/NT
and finding DF+DFD reads as either not understanding the benchmark or misrepresenting it. This
was caught and named honestly rather than left to surface later.

**Consequence for LOMO**: with only two manipulation families present, the leave-one-manipulation-out
axis is a **2-config comparison** (train Deepfakes → test DeepFakeDetection; train
DeepFakeDetection → test Deepfakes), not the four-way comparison the "LOMO" name might suggest.
Still a legitimate unseen-manipulation test — DF (classic autoencoder swap) and DFD (Google's
actors, different generation pipeline) come from genuinely different methods — but it's
supporting evidence for cross-dataset generalization, not the primary claim. The primary
generalization evidence is the real cross-dataset tests (Celeb-DF official test split, DFDC
when added), which stay unaffected by this.

**Class balance, computed from real disk counts** (not assumed from FF++'s nominal 1:4
structure): 1,363 real / 4,066 fake across this FF-derived set = **25.11% real**. To reach
~1:1 balance via clip-count sampling (`data/splits.py:balance_by_clip_sampling`) at 1 clip per
fake video, real videos need **~2.98 clips/video** (1,363 × 2.98 ≈ 4,062 real clips vs 4,066
fake clips). Recompute this ratio if the corpus composition ever changes — it is not a fixed
constant, it was derived from the counts above on this date. Celeb-DF-v2's real videos are
**not** added to training to close this gap — Celeb-DF stays test-only, per the dataset-roles
table in `docs/RESEARCH_PLAN.md`; balance the class ratio via clip sampling, not by mixing in
another dataset's real videos.

## Metric definitions

| Metric | Computed in | Convention-sensitive? |
|---|---|---|
| Accuracy | `train/eval_predictions_common.py` | No — symmetric |
| ROC-AUC | `train/eval_predictions_common.py` | No — symmetric |
| F1 / Precision / Recall | `train/eval_predictions_common.py` | **Yes** — see above |
| Confusion matrix | `train/eval_predictions_common.py`, ordered `[real, fake]` (labels `[1, 0]`) | Ordering is fixed, not convention-dependent |

Evaluation is **per-sample** by default (one row per frame/clip, not aggregated across a video) — see
"Multi-clip inference" below for the one place that changes.

## Split policy

Two split engines exist in `data/dataloader.py:DatasetBuilder`, selected via `split_engine=`:

### `"legacy"` (default — unchanged behavior for every existing completed run)

- **Images**: preserves the source dataset's own train/test folders; validation carved from source
  training data (85/15). Not identity-aware — no meaningful "identity" concept for these datasets.
- **Video/frame**: identity-aware grouped split (70/10/20), using `DatasetBuilder.get_identity`'s two
  hardcoded regex heuristics. Falls back to treating an unmatched filename as its own unique identity —
  known limitation, not fixed in this path on purpose (see `"identity_v2"` below).
- Balancing: train-split-only oversampling (`_apply_class_balance`), repeats the minority class.
- **Verified**: reproduces the exact historical split counts for at least one completed run
  (`VID-SPA-02`: 14096/866/2448, train balanced 7048/7048) — this is the regression gate that must never
  break.

### `"identity_v2"` (opt-in — corrected, NOT backward-compatible in split composition)

- Identity extraction: `data/identity.py`, explicit per-dataset regex strategies. **Fails loudly**
  (`IdentityExtractionError`/`KeyError`) on a filename that doesn't match its dataset's declared pattern,
  or a dataset with no declared strategy at all — no silent per-file fallback.
- Multi-identity grouping: `data/splits.py` union-find over all identities appearing in a record (a
  face-swap video legitimately involves two identities — source and target — and both must stay on the
  same side of a split, even when only one of them is mentioned in a *different* record).
- **Celeb-DF-v2 special case**: honors the dataset's official published test list
  (`datasets/videos/celeb-df-v2/List_of_testing_videos.txt`, 178 real / 340 fake) instead of running the
  generic identity split on it. Reason: Celeb-DF-v2's fake videos densely cross-link identities into a
  few giant connected components, so a naive identity split can produce a 0%-real eval split — verified
  empirically. `check_leakage()` is intentionally NOT enforced across the official-test boundary (that's
  the accepted community benchmark protocol), but IS enforced within the train/val remainder.
- **FaceForensics++ manipulation tagging**: `data/identity.py:detect_ffpp_manipulation` tags each record
  with its manipulation type, required for LOMO (below). This repo's on-disk FF++ copy currently has only
  `Deepfakes` and `DeepFakeDetection` fake subsets — NOT the full 4-manipulation release
  (`Face2Face`/`FaceSwap`/`NeuralTextures` aren't downloaded here yet).
- Balancing: asymmetric clip-count sampling (`balance_by_clip_sampling`) — repeats each video's path
  `n_clips` times in the sample list rather than oversampling whole records; the minority class gets more
  clips per video. **Convention check**: `n_real`/`n_fake` and all real/fake branching in `data/splits.py`
  use `label==1` for real — this was backwards once (inherited `fake=1` from `docs/code/splits.py` during
  promotion) and got caught and fixed; if you touch this file, verify against this document, not memory.
- `check_leakage()` is a hard, non-bypassable `RuntimeError` on any detected leak (except the deliberate
  Celeb-DF official-test exception above) — a split that would leak never gets returned to a caller.

### Leave-one-manipulation-out (LOMO)

`data/splits.py:usable_lomo_configs()` + `DatasetBuilder.prepare_lomo_split()`. Filters
`lomo_configs()`'s nominal 4-manipulation list down to what's actually present in the corpus, AND rebuilds
`train_manipulations` from the present set rather than trusting the nominal list — a real bug was caught
here during development where the nominal list silently excluded every `DeepFakeDetection` fake video from
training (not held out, not tested, just dropped) because `DeepFakeDetection` was never one of the 4
standard names. `prepare_lomo_split()` re-derives `train_manipulations` internally regardless of what's
passed in, so this can't recur even from a caller mistake.

## Clip-sampling semantics

- `mode="single"`: one sampled frame per video. `clip_sampling="random"` (train) picks a random frame index
  per `__getitem__` call; `"center"` (eval) always picks the middle frame.
- `mode="sequence"`: a contiguous clip of `seq_len` frames, spaced by `stride` (default `1` — every
  completed run used this; verified the stride-aware `_contiguous_indices` collapses to byte-identical
  output when `stride=1`). `stride>1` widens temporal coverage (e.g. `seq_len=16, stride=4` spans 64
  source frames). Train: random start offset. Eval: centered.
- **Multi-clip inference** (`DatasetBuilder.build_multi_clip_eval_samples`, wired into
  `train/video/test_video.py` via `--n-clips`): repeats a video's sample N times with `clip_sampling="random"`
  (not `"center"` — repeating a centered clip N times would just be the same clip N times), then aggregates
  per-video via a mean of `prob_real` across that video's clips before computing final metrics. Default
  `--n-clips 1` is byte-identical to the original single-clip path.

## Decode-failure policy

`DeepFakeDataset(decode_failure_policy=...)`, values: `"legacy_zero"` (default — silent zero-tensor with
the original label, exactly the pre-refactor behavior) or `"raise"` (re-raises the decode exception).
A third option, manifest-driven `"drop"`, is implemented at the `DatasetBuilder` level instead
(`manifest_path=` — see `proc/integrity_scan.py`/`proc/manifest.py`), since dropping needs a
pre-computed validity list, not a per-`__getitem__` decision.

**T1.0 audit result** (`python -m train.video.validate_video_dataset --dataset-scope video_all --mode both
--decode-backend cv2`): **zero decode failures** across the entire 12,028-video corpus, both `single` and
`sequence` modes, under `cv2` — the backend every completed run actually used. The historical 22 completed
runs are not contaminated by fabricated zero-tensor samples. The default has not been changed from
`"legacy_zero"` yet regardless — that's still an open call, not automatically resolved by a clean audit.

## Seed handling

`--seed` CLI flag on both `run_image.py`/`run_video.py`, default `42` (unchanged). Run-directory naming:
suffixed with `_seed{N}` **only when the seed differs from the default** — so every existing completed-run
path stays byte-identical, and a non-default seed no longer silently overwrites the default run's
checkpoint directory. Same pattern applied to `stride` (`_stride{N}`, only when `!=1`).
