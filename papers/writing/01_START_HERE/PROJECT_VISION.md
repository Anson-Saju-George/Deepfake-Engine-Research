# Project Vision — Cross-Regime Generalization in Video Authenticity Detection

**Status:** design frozen Sept 2026 · experiments not yet started
**Prior attempt:** rejected by TCSST, July 2026
**Constraint:** results needed to support a graduate application, Dec 2026 – Jan 2027

---

## 1. The problem

A video arrives. Is it authentic, or was it produced or altered by a generative model?

That single question now spans two technically unrelated phenomena:

| Regime | What "fake" means | Where the evidence lives |
|---|---|---|
| **Face manipulation** | A real recording — real scene, real body, real background — with the **face** swapped or reenacted | A blending seam, warp, or texture discontinuity in a small facial region |
| **Fully synthetic** | The **entire video** generated from a text or image prompt; no real source exists | Whole-frame: implausible physics, texture statistics, temporal incoherence |

The field grew up on the first and is racing to absorb the second. Detectors, datasets,
benchmarks and preprocessing conventions were all built around face manipulation — most
pipelines literally crop to a detected face and discard the rest of the frame. That
convention is now in tension with a fake type that often contains no face at all.

The obvious question — *can one detector do both?* — has been answered affirmatively
(UNITE, CVPR 2025). The unobvious and unanswered question is **what actually transfers**.

## 2. The research question

> Do forensic representations learned for **partial facial manipulation** transfer to
> **fully synthetic video**, does the reverse transfer exist, and does joint training
> produce genuinely regime-invariant evidence — or merely dataset discrimination?

Framed as a hypothesis, not a claim. The experiment decides.

**Three sub-questions:**

- **RQ1 — Cross-regime transfer.** Face → synthetic, and synthetic → face. Is transfer
  symmetric? (Prior: it will not be.)
- **RQ2 — Unified training.** Does regime-balanced joint training beat the specialists on
  their own domains while holding cross-domain performance — and does it unify the
  representation, or just learn two decision regions?
- **RQ3 — Audited generalization.** Do RQ1/RQ2 survive removal of dataset and source
  shortcuts?

## 3. Why this is defensible

Three adjacent results already exist. Positioning matters:

| Already claimed | By | What it leaves open |
|---|---|---|
| Unified whole-frame detection *works* | UNITE, CVPR 2025 | Trained FF++ vs FF++ + **GTA-V game footage** — not AI-generated video — and never on the GenVideo split. No bidirectional decomposition. |
| Real / partially-manipulated / fully-synthetic is a valid taxonomy | Omni-Fake, CVPR 2026 | Formalises the categories; does not measure transfer between them. |
| Benchmark shortcuts inflate reported generalization | VidAudit, 2026 | Provides the audit protocol; does not apply it to the cross-regime question. |

**Our contribution is the intersection none of them occupy:** the controlled,
**bidirectional** train-on-one-regime / test-on-the-other decomposition, executed under an
audited protocol, with regime-balanced joint training as the third condition.

Supporting evidence that the question is live: UNITE reports **FF++ → DeMamba 57.38 AUC**,
rising to **93.75** once an unrelated synthetic domain is added to training. Something
transfers, and something does not. Nobody has decomposed which.

## 4. Technical approach

### 4.1 Pipeline — whole-frame, no face detection

```
video → decode all frames (native fps) → resize whole frame → WebDataset shard cache
      → 16-frame clips, stride 1 → backbone → mean-pool → binary real/fake
      → non-overlapping tiling at eval → averaged video-level score
```

Deliberately **no face detector, no face crop.** Two reasons: fully synthetic videos often
contain no face, and UNITE demonstrates whole-frame beats face-crop even on face-manipulation
benchmarks. The cost is real and accepted — SBI-style face-blending augmentation becomes
unavailable, and several modern datasets (DF40, KoDF) ship pre-cropped and are therefore
unusable.

Resolution **384×384** (matching UNITE), confirmed by a 256/384/512 pilot with a
predetermined adoption rule: take the smallest resolution within ~0.5–1 AUC of the best.

### 4.2 Corpus

| Regime | Train | In-domain test | OOD test |
|---|---|---|---|
| Face manipulation | FF++ c23 (4 manipulations) | FF++ held-out | DeeperForensics, Celeb-DF |
| Fully synthetic | GenVideo (subsampled) | GenVideo held-out | GenVideo OOD, DeepAction |
| Unified | balanced FF++ + GenVideo | both | all external |

**Two structural rules that govern everything:**

1. **Source-matched reals.** Never "fakes from dataset A vs random reals from dataset B."
   Where matching is impossible, report matched-real and cross-domain-real evaluations
   separately, never pooled.
2. **Scale-controlled comparison.** FF++ has ~5,000 videos; GenVideo has ~2.26M. Balanced
   *sampling* across a 450× gap means each FF++ video is revisited hundreds of times per
   epoch — memorisation, not training. GenVideo is subsampled to FF++ scale for the
   controlled comparison; full scale is a separate ablation.

### 4.3 Models — two backbones, no zoo

- **ConvNeXt-B** — supervised CNN baseline
- **SigLIP-So400M** (or DINOv2) — foundation encoder; UNITE uses SigLIP, and without a
  foundation arm any performance gap will be attributed to the backbone rather than the
  training regime

The independent variable is **training regime**, not architecture. The earlier
temporal-head sweep (TCN / LSTM / Transformer / ConvLSTM) is cut.

### 4.4 The core experiment

```
                    FACE                          SYNTHETIC
             FF++  DeeperF.  CelebDF       GenVideo-OOD  DeepAction
F  Face-only    ✓       ✓        ✓               ★            ★
S  Synth-only   ★       ★        ★               ✓            ✓
U  Unified      ✓       ✓        ✓               ✓            ✓
```

★ marks the contribution. Two backbones × three regimes × five test domains, plus audits.

### 4.5 Audits — the third of the paper that makes the rest trustworthy

A three-feature **clip-length** classifier reaches 0.998 AUC on a standard AI-video
benchmark while measuring nothing about motion; under audit it collapses to chance. A
20-paper survey found none applying the full control set. This is the field's central
methodological weakness, and any number produced without controls is suspect.

**Adopted from VidAudit** (cited, not claimed): canonical codec re-encoding · clip-length
and leakage filtering · real-vs-real dataset identity probe · matched retraining ·
multi-seed + bootstrap CIs · true cross-dataset evaluation.

**Our complementary diagnostic:** a 32×32 thumbnail probe — destroy fine forensic detail,
retain composition, codec and style. If classification survives, dataset signature is doing
the work.

**Elevated to a preprocessing rule, not an audit:** fixed-duration clips and canonical
re-encoding across every dataset, applied *before* training. GenVideo fakes are short
generated clips; FF++ reals are ~15 s YouTube videos. Duration alone may separate the
regimes, and no post-hoc audit repairs a model that already learned it.

### 4.6 Metrics

VidAudit found that at a deployable FPR of 0.1%, high-AUC methods collapse to single-digit
recall *and the leaderboard order changes*. AUC alone is insufficient. Report the audited
tuple: **AUC · above-floor margin · operating-point recall @ FPR 0.1% · calibration.**

Per-dataset always, never pooled. Threshold frozen on training-domain validation. Bootstrap
CIs *and* 3 seeds with mean±std and Wilcoxon — the prior rejection cited absent repeated-run
analysis, and bootstrap alone does not answer it.

## 5. What already exists

Substantial infrastructure is built and benchmarked:

| Component | State |
|---|---|
| Repository | Audited and refactored (6-task programme complete) |
| Data pipeline | Centralised, dataset registry, identity-aware splits with union-find grouping, leakage checker |
| Cache | WebDataset tar-shards, per-key reader with persisted offset index |
| Environment | `/opt/ml`, torch 2.11.0+cu129, cuDNN 92500, RTX 5080 (16.3 GB), verified clean |
| Benchmarks | pyav decode 7.5× over cv2 · `num_workers=12` · AMP **bf16** +56% · cache on F: with dataloader wait **0.25 ms** (GPU-bound at 76–86% util) · seq16/batch4 = **13.56 clips/sec** · batch ≥12 silently spills to system RAM |
| Corpus | Celeb-DF extracted and cached (30.36 GB); FF++ partially on disk; DeeperForensics access approved |
| Prior runs | 22 archived under the old face-crop protocol — reference only, not evidence |

**The gap:** the pipeline was built for face crops. `extract_frames.py` — the whole-frame,
dataset-agnostic extractor — does not exist yet. That is the critical path.

## 6. Why the previous attempt failed

TCSST rejected three things, and only one was a writing problem:

1. **Redundancy** — the same three claims restated across related work, results, discussion
   and conclusion
2. **No generalization evaluation** — no cross-dataset, no unseen-generator, no robustness
   testing
3. **No statistical rigour** — single seed, no mean±std, no significance tests

The root cause of (2) was structural: the old work **trained on Celeb-DF and FF++ combined**,
leaving no genuine cross-dataset test set. The new design fixes this at the corpus level —
one training source per regime, everything else held out — rather than by adding text.

## 7. Both outcomes are publishable

| If the matrix shows | The paper is |
|---|---|
| F→F 0.95 · F→S 0.61 · S→S 0.93 · S→F 0.58 · U→both 0.82–0.86 | Specialists work; cross-regime transfer collapses; naive unification trades away specialisation |
| U→both ≈ 0.94 | Despite severe cross-regime transfer failure, joint training learns a representation spanning manipulation and synthesis |

The design is productive either way. **The matrix is the experiment that determines what
paper this is — it runs before anything expensive.**

## 8. Execution

**Phase A — validate the protocol.** ConvNeXt-B, resolution pilot, one face OOD, one
synthetic OOD, source-matched reals throughout.

**Phase B — answer the question.** The five-cell matrix.

**Phase C — defend it.** Thumbnail probe · real-vs-real source probe · dataset-identity
probe · second backbone · balanced-vs-natural sampling ablation.

**Extraction order:** write `extract_frames.py` first (dataset-agnostic, resolution-agnostic,
full metadata), then FF++ → GenVideo (subsampled) → DeeperForensics → Celeb-DF → DeepAction.
Do not preprocess hundreds of gigabytes before the design is validated.

**Venue:** deferred. The shape of the transfer matrix determines what kind of paper this is.

---

## 9. Open decisions

1. **GenVideo extraction budget** — 2.26M videos is infeasible; ~10–20k proposed for the
   controlled comparison plus the full OOD split
2. **Second backbone** — SigLIP-So400M vs DINOv2
3. **Omni-Fake verification** — unconfirmed; load-bearing for the novelty argument
4. **Whether the unified condition needs its own source control** — with FF++ matched
   (YouTube real / face fake) and GenVideo matched (Kinetics real / synthetic fake), source
   maps onto regime in the unified arm and can be learned instead of forgery
