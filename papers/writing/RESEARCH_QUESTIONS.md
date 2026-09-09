# Research Questions — the scientific contribution

The contribution is structured as a **progression of three questions**, not as another
universal detector. Each question is designed so that a *negative* result is still a finding.
For how each is operationalised (datasets, splits, conditions, metrics), see
`EXPERIMENT_DESIGN.md`; for why it matters, see `PROJECT_VISION.md`.

The two regimes throughout:
- **Face manipulation** — a real recording with the face swapped/reenacted (evidence is a local blending seam/warp). Trained on **FF++**.
- **Fully synthetic** — the entire video generated from a prompt (evidence is whole-frame: physics, texture statistics, temporal incoherence). Trained on **GenVideo**.

---

## RQ1 — Cross-regime transfer

**Does a detector trained on one regime transfer to the other, and is the transfer symmetric?**

Two directed questions plus a symmetry question:
1. **Face manipulation → fully synthetic:** does an FF++-trained detector detect fully synthetic video?
2. **Fully synthetic → face manipulation:** does a GenVideo-trained detector detect face manipulation?
3. **Is the transfer symmetric?** i.e. is `F→S ≈ S→F`?

**Hypothesis:** the transfer is **not symmetric**. Face-manipulation detectors key on a
localized facial artifact that fully synthetic video need not contain, while
synthetic-video detectors key on whole-frame statistics that a real-background faceswap
largely preserves — so one direction is expected to collapse harder than the other. The
asymmetry itself is the interesting result: it says the two regimes are not interchangeable
and that "deepfake detection" is not one task.

> This is the core matrix (the ★ off-diagonal cells in `EXPERIMENT_DESIGN.md §2`). Always
> read each transfer cell against its own within-regime diagonal — regime difficulty is not
> equal, so a raw `F→S` vs `S→F` comparison is only meaningful relative to `S→S` and `F→F`.

---

## RQ2 — Unified training

**Does a detector trained on `FF++ + GenVideo` beat the specialists on their own domains
while maintaining cross-domain performance — and does it genuinely *unify* the forensic
representation, or merely learn two separate decision regions?**

Two parts, and the second is the deeper one:
1. **Performance:** does the unified model match/beat `F` on face and `S` on synthetic, *and*
   hold up cross-domain (i.e. close the RQ1 transfer gaps)?
2. **Mechanism:** *how* does it succeed if it does? A model can post strong unified numbers by
   internally partitioning into a face-detector and a synthetic-detector stitched behind one
   head — that is **not** a unified forensic representation.

**Probe the mechanism, don't infer it from accuracy.** Examine the learned embeddings /
run domain probes:
- Does a linear probe on the frozen representation predict **regime/dataset** more easily than **real/fake**? If regime is trivially decodable, the representation is partitioned, not unified.
- Does unified training make dataset/regime identity **more or less** linearly separable than the specialists do?
- Do face and synthetic fakes occupy one shared "fake" region, or two disjoint clusters?

A unified *number* with a partitioned *representation* is a weaker result than it looks, and
saying so honestly is part of the contribution.

---

## RQ3 — Audited generalization

**Do the RQ1 and RQ2 findings survive the removal of dataset/source shortcuts?**

Any cross-regime or unified result is only credible if it is not an artifact of a dataset
signature (codec, clip length, real-source provenance, resolution). Re-run the RQ1/RQ2
conclusions under controls:
- **VidAudit-style controls** (adopted, cited, not claimed as novel): canonical codec re-encoding, clip-length/leakage filtering, real-vs-real source-identity probe, matched retraining, multi-seed + bootstrap CIs, true cross-dataset evaluation.
- **Thumbnail diagnostic** (our complementary probe): destroy fine forensic detail (e.g. 32×32) while retaining composition/codec/style — if classification survives, dataset signature is doing the work, not fakery.

The question RQ3 answers: **which of the RQ1/RQ2 effects are real forensic generalization,
and which are shortcuts?** An effect that vanishes under audit is itself a reportable
finding about the benchmark, not a failure.

---

## Why this progression

| Instead of | This asks |
|---|---|
| "We built another universal detector." | *Are these even the same task?* (RQ1) → *Can one model truly unify them, mechanistically?* (RQ2) → *Is any of it real once shortcuts are removed?* (RQ3) |

Each RQ is publishable in either outcome: asymmetric/collapsing transfer, partitioned-vs-unified
representations, and survive-vs-vanish-under-audit are all findings. The matrix decides which
paper this is — run it before anything expensive.
