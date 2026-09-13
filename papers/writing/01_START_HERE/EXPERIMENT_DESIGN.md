# Experiment Design — frozen Sept 2026

**Thesis (hypothesis, not assertion — the matrix decides):**

> Face manipulation and fully synthetic video are related but non-identical forensic
> regimes. We quantify **bidirectional** cross-regime transfer, test whether balanced
> joint training yields genuinely regime-invariant evidence, and audit the result against
> dataset-specific shortcuts.

**What is already claimed by others (do NOT claim these):**
- Unified whole-frame detection *works* → UNITE, CVPR 2025
- Real / partially-manipulated / fully-synthetic is a valid taxonomy → Omni-Fake, CVPR 2026
- Dataset shortcuts inflate AI-video benchmarks; real-vs-real source probe → VidAudit, 2026

**What remains open (our contribution):** the controlled, bidirectional
train-on-one-regime / test-on-the-other decomposition, run under an audited protocol,
with regime-balanced joint training as the third condition.

---

## 1. Dataset roles — FROZEN

| Regime | TRAIN | In-domain test | OOD test |
|---|---|---|---|
| **Face manipulation** | FF++ (c23, 4 manipulations) | FF++ held-out | DeeperForensics, Celeb-DF |
| **Fully synthetic** | GenVideo train split | GenVideo held-out | GenVideo OOD, DeepAction |
| **Unified** | balanced FF++ + GenVideo | both | all external sets |

**Changes from the earlier plan:**
- **GenVideo is now the synthetic TRAIN corpus** (was DeepAction).
- **DeepAction returns to TEST-ONLY** — a fully independent external synthetic test.
- **ForgeryNet: DROPPED from core experiments.** 496 GB adds preprocessing faster than it
  adds clarity. Reserve only if a reviewer demands another partial-manipulation domain.
- **Old 22 runs: archived, not designed around.** At most rerun 1–2 under the new splits
  as a "face-centric baseline" for a *Why we moved beyond face crops* subsection.

### ⚠️ Leakage rules (non-negotiable)
1. **DeeperForensics ships no reals** — its detection real class is the 1,000 FF++ YouTube
   target videos. **Split by FF++ target ID first**, then use only held-out FF++ originals
   with the matching DF-1.0 manipulations at test time.
2. **Never construct a condition as "fakes from dataset A vs random reals from dataset B."**
   Pair reals to fakes from the same source/domain wherever possible.
3. Where matched reals are impossible, report **matched-real** and **cross-domain-real**
   evaluations *separately*. Never silently mix them.

---

## 2. Training conditions — three, and only three

| ID | Condition | Training data |
|---|---|---|
| **F** | Face specialist | FF++ only |
| **S** | Synthetic specialist | GenVideo only |
| **U** | Unified | regime-balanced FF++ + GenVideo |

### The core table (★ = the contribution)

```
                    FACE                          SYNTHETIC
             FF++  DeeperF.  CelebDF       GenVideo-OOD  DeepAction
F  Face-only    ✓       ✓        ✓               ★            ★
S  Synth-only   ★       ★        ★               ✓            ✓
U  Unified      ✓       ✓        ✓               ✓            ✓
```

### ⚠️ Regime-balanced sampling (critical)
FF++ has ~5,000 videos; GenVideo train has ~2.26M. Naive concatenation makes "unified"
into "a GenVideo detector that occasionally sees FF++." Enforce per batch:

- 25% FF++ real · 25% FF++ manipulated · 25% GenVideo real · 25% GenVideo synthetic

**Additional control I require beyond balanced sampling:** balanced *sampling* over a
450× size gap means every FF++ video is revisited ~450 times per GenVideo epoch — that is
memorisation, not training. **Subsample GenVideo to FF++ scale (~10–20k videos) for the
controlled F/S/U comparison**, and run full-scale GenVideo separately as a scale ablation.
Otherwise the three conditions differ in data volume as well as regime, and the comparison
is not controlled.

Ablate natural-frequency sampling separately.

---

## 3. Preprocessing — decided

| Parameter | Value |
|---|---|
| Pipeline | **whole-frame, no face detection, no face cropping** |
| Resolution | **384×384 default** (UNITE uses 384). Run the 256/384/512 pilot to confirm. |
| Pilot rule | Identical samples, seed, and optimizer-step count at each resolution. Adopt the **smallest** resolution within ~0.5–1 AUC of the best. |
| Cache | Extract once at sufficient resolution; resize in the dataloader. Do **not** build three permanent caches. |
| Sampling | native fps, stride 1, seq_len 16 |
| Cache location | F: (drvfs) — benchmarked, dataloader wait ~0.25 ms, GPU-bound |
| Precision | AMP bf16 + channels_last |
| Workers | 12 |

### ⚠️ Duration + codec normalisation is a PREPROCESSING rule, not an audit
VidAudit's clip-length classifier reached 0.998 AUC measuring nothing but duration.
GenVideo fakes are short generated clips; FF++ reals are ~15 s YouTube videos. **Duration
alone may separate the regimes.** Therefore, before any training:
- fixed-duration clips across every dataset
- canonical codec re-encode for every source
- record original duration/codec/bitrate in metadata so the confound can be *tested*, not
  assumed away

---

## 4. Models — two backbones, no zoo

| Arm | Model | Why |
|---|---|---|
| Supervised CNN | **ConvNeXt-B** | strong conventional baseline, already benchmarked here |
| Foundation encoder | **SigLIP-So400M** (or DINOv2) | UNITE uses SigLIP-So400M; without a foundation arm your numbers will sit far below published ones and the gap will be attributed to the backbone |

Video aggregation: sampled-frame embeddings/logits → mean pooling.
**The temporal-head sweep (TCN / LSTM / Transformer / ConvLSTM) is CUT.** The independent
variable is now *training regime*, not architecture. Add at most one temporal ablation if
frame aggregation leaves an obvious question open.

---

## 5. Metrics — revised per VidAudit

VidAudit found that at a deployable FPR of 0.1%, multiple high-AUC methods fall to
single-digit recall **and the leaderboard order changes**. AUC alone is therefore not
sufficient. Report the **audited tuple**:

| Metric | Role |
|---|---|
| **Video-level AUROC** | primary |
| **Above-floor margin** | vs the trivial/shortcut baseline |
| **Operating-point recall @ fixed FPR (0.1%)** | deployability |
| **Calibration** | reliability |
| AUPRC, balanced accuracy, F1 | secondary |

Reporting rules:
- **Every dataset separately + macro-average. Never a pooled headline number.**
- Threshold chosen on the training-domain validation set, then **frozen** for all external tests.
- Video-level **bootstrap 95% CIs** — and, separately, **3 seeds with mean±std and Wilcoxon**
  on claim-carrying comparisons. Bootstrap is within-run uncertainty; it does not answer
  the prior rejection's "no repeated-run analysis / significance testing" objection. Both
  are required.

### ⚠️ Regime difficulty is not equal
GenVideo synthetic detection may simply be easier than FF++ face manipulation. If
S→S = 0.99 while F→F = 0.95, the transfer cells are not comparable across rows. Always
report the within-regime diagonal as the reference for its own row.

---

## 6. Audits — adopted from VidAudit, plus one diagnostic

**Adopt (cite VidAudit, do not claim as novel):**
1. canonical codec re-encoding
2. clip-length / leakage filtering
3. **real-vs-real dataset identity probe** — can a classifier tell FF++/YouTube reals from
   GenVideo/Kinetics reals? If yes, the source signature is strong and must be controlled
4. matched retraining
5. multi-seed + bootstrap CIs
6. true cross-dataset evaluation

**Our complementary addition (diagnostic, not Contribution #2):**
- **32×32 thumbnail probe** — destroy fine forensic detail, retain composition/codec/style.
  If classification survives, dataset signature is doing the work. Answers a different
  question from the real-vs-real probe and is worth running alongside it.

**Also probe:** does the frozen representation predict *dataset* better than *real/fake*?
Does unified training make dataset identity more or less separable? There is 2026 work
showing domain differences can dominate the real/fake distinction in multi-domain
face-forgery detection.

---

## 7. Extraction architecture — adapters + a shared core

Each dataset is **wired differently** — directory layout, filename→identity/label mapping,
how real vs fake is signalled, whether reals are in-set or borrowed from another dataset,
native fps/resolution quirks. But the *actual* frame extraction and preprocessing is identical
across all of them. So the extractor is **two layers, not one monolith**:

**(a) Per-dataset adapter layer — the only dataset-specific code.** One small adapter per
dataset whose sole job is to walk that dataset's real on-disk structure and emit a stream of
**normalized `VideoRecord`s** conforming to the schema below. Each adapter owns:
- enumerating video files under that dataset's layout
- parsing filename/path → `identity`, `label`, `regime`, `manipulation_method`/`generator`, `split`
- the real/fake rule for that dataset (and, where reals are borrowed, the `source_real_dataset` linkage — e.g. DeeperForensics reals = FF++ YouTube originals matched by target id, per §1's leakage rule)
- any per-dataset fps / container / resolution normalization hints

Adapters are written **after the dataset is on disk**, against its actual structure — not
guessed up front.

**(b) Shared core — dataset-agnostic, written once, reused by every adapter.** Takes a
`VideoRecord` and does the invariant work:
- decode (CPU `pyav` per `perf/PERFORMANCE_LOG.md`), **whole-frame, no face detection/cropping**
- **duration + canonical codec re-encode** and resolution normalization (§3 — a hard preprocessing rule, so clip-length/codec can't become the shortcut VidAudit found)
- frame sampling (native fps, stride, seq_len) and resize in the dataloader
- write frames + a per-frame metadata row

This keeps ~90% of the logic (the hard, correctness-critical part) in one tested place; adding
a new dataset = writing one thin adapter, not touching the core.

**Metadata schema — recorded per video/frame by the core, populated by the adapter:**

```
video_id · dataset · regime · label · source_real_dataset · manipulation_method ·
generator · split · compression · resolution · fps · duration · frame_index ·
original_resolution
```

Without `dataset`, `regime`, and `source_real_dataset` as explicit variables, the shortcut
audits in §6 cannot be run at all.

---

## 8. Execution order

**Phase A — validate the protocol (small)**
ConvNeXt-B · 256/384/512 pilot · FF++ train · GenVideo train · one face OOD · one
synthetic OOD · source-matched reals throughout.

**Phase B — answer the question**
F→Face, F→Synthetic, S→Synthetic, S→Face, U→all. That is the core matrix.

**Phase C — defend it**
Thumbnail probe · real-vs-real source probe · dataset-identity probe · second backbone
(SigLIP) · balanced-vs-natural sampling ablation · optionally one temporal ablation.

**Extraction order (per §7 architecture):**
1. **Pull the dataset first**, then inspect its actual on-disk wiring.
2. Write/verify that dataset's **thin adapter** against the real structure (emits normalized `VideoRecord`s).
3. Run the **shared core** (decode → codec/duration normalize → sample → metadata). The core is built once, up front; only adapters are added per dataset.

Dataset order: FF++ → GenVideo (subsampled) → DeeperForensics → Celeb-DF → DeepAction.
**Do not preprocess 500 GB before the experimental design is validated** — validate the
core + first adapters on a small slice, then scale.

**Scope decision still open:** how many GenVideo videos to actually extract. 2.26M is not
feasible here. Propose ~10–20k for the controlled comparison, plus the full OOD test split.

**Venue:** deferred until the transfer matrix exists. Its shape determines what paper this is.

---

## 9. Both outcomes are publishable

| If the matrix shows | The paper is |
|---|---|
| F→F 0.95, F→S 0.61, S→S 0.93, S→F 0.58, U→both 0.82–0.86 | Specialists work; cross-regime transfer collapses; naive unification trades away specialisation. |
| U→both ≈ 0.94 | Despite severe cross-regime transfer failure, joint training learns a representation spanning manipulation and synthesis. |

The design is productive either way. **The matrix is the experiment that tells you what
paper you have — run it before anything expensive.**
