# Deepfake Detection — Full Programme (40 hr/week strategy)

**Start:** 21 Jul 2026 · **Paper submit target:** ~10–12 Sep 2026 (+ arXiv same day)
**Thesis:** through May 2027

**Principle:** frame-level artifacts carry most of the signal; aggregation over more
frames beats heavier temporal heads. Priority: frame encoder → sampling/aggregation
→ temporal head.

**Key scheduling fact:** each 10-epoch run is ~1.5–2 GPU hours. The full programme is
~35 runs ≈ 70–120 GPU hours, absorbed overnight. The schedule is bound by HUMAN hours,
so writing runs in parallel with experiments from Week 2.

---

## Master phase table

| # | Phase | Weeks | Hrs | Runs | Gate (measurable) |
|---|---|---|---|---|---|
| 0 | Free wins | W1 | 30 | 0 | 36-cell robustness table + multi-clip delta |
| 1 | Corpus rebuild | W2 | 40 | caching | check_leakage()=NONE, clip %real in [45,55] |
| 2 | Tier A aggregation | W3 | 40 | 4 | best-A > A0 by >=0.02 AUC |
| 3 | Tier B SBI + generalization | W4–5 | 80 | 9 | B1 > B0 by >=0.05 AUC on Celeb-DF |
| 4 | Tier C temporal | W6 | 40 | 7 | 6 heads @16 frames; C3 answered |
| 5 | Statistics | W7 | 40 | 15 | every claim has mean±std + Wilcoxon p |
| — | Assembly + submit | W8 | 40 | 0 | SUBMITTED + arXiv |
| 6 | Temporal decay | Sep–Nov | — | ~10 | decay curve, 4 dataset generations |
| 7 | Audio-visual | Dec–Mar | — | ~15 | AV consistency chapter |
| — | Thesis assembly | Mar–May 27 | — | 0 | thesis submitted |

---

## Week-by-week

| Wk | Dates | Daytime | Overnight | Writing |
|---|---|---|---|---|
| 1 | Jul 21–27 | Phase 0, integrity, access forms | face caching | — |
| 2 | Jul 28–Aug 3 | Corpus rebuild, leakage gate | caching → A-runs | Methods §3–4 |
| 3 | Aug 4–10 | Tier A (A0–A4) | A-runs | Related Work §2 |
| 4 | Aug 11–17 | SBI impl + B0/B1 | B-runs | Setup §5–6 |
| 5 | Aug 18–24 | B2–B4, cross-dataset, LOMO | B/LOMO runs | Intro §1 |
| 6 | Aug 25–31 | Tier C heads @16 frames | C-runs | Results §7 |
| 7 | Sep 1–7 | 3 seeds, stats, tables/figures | seed runs | Discussion §8 |
| 8 | Sep 8–14 | Assembly, review, polish | — | SUBMIT + arXiv |

---

## Experiment matrix

### Tier A — aggregation (W3)
| ID | Config |
|---|---|
| A0 | ConvNeXt-B, 4-frame centre clip (baseline for the delta) |
| A1 | ConvNeXt-B, 16 frames stride 4, mean pool |
| A2 | ConvNeXt-B, top-k mean (k=5) |
| A3 | ConvNeXt-B, gated attention MIL |
| A4 | dense multi-clip inference (inference-only, applies to all) |

### Tier B — generalization (W4–5)
| ID | Config |
|---|---|
| B0 | best-A trained normally (cross-dataset baseline) |
| B1 | best-A + SBI (real-only training) |
| B2 | best-A + frequency branch (DCT/SRM) |
| B3 | best-A + compression augmentation |
| B4 | B1 + B3 |

Evaluate every row on: FF++ test · LOMO (unseen manipulation) · Celeb-DF official
test · DFDC-Preview.

### Tier C — temporal (W6)
| ID | Config |
|---|---|
| C1 | ConvNeXt-L + TCN @16 frames |
| C2 | + LSTM, Transformer, ConvLSTM, hybrid-TCN, hybrid-Transformer |
| C3 | best temporal + SBI — does temporal add anything once frame features are strong? |

---

## Dataset roles (never violate)

| Dataset | Role |
|---|---|
| FF++ c23 (all 4 manipulations) | TRAIN only |
| Celeb-DF v2 official test | TEST only, permanently |
| DFDC-Preview | TEST only, permanently |
| DeeperForensics-1.0 | TEST only (unseen manipulation + robustness) |
| KoDF / DeepSpeak / ForgeryNet public test | TEST only, Phase 6 |

---

## Reviewer-objection traceability

| Objection | Addressed by |
|---|---|
| #1 redundancy | each theme stated ONCE in Results, referenced by §number elsewhere |
| #2 cross-dataset / unseen generator / robustness | Phase 3 (cross-dataset), Phase 1+3 (LOMO), Phase 0 (perturbations) |
| #3 repeated runs / std / significance | Phase 5 (3 seeds, mean±std, Wilcoxon) + bootstrap CIs |

---

## Reporting rules (do not break)

1. AUC is the PRIMARY metric — balance-invariant, matches the literature.
2. Always state which class is positive for F1; report both.
3. Report test-set composition (n_real / n_fake / % real) in every table.
4. Never train on a test-only dataset.
5. Fix thresholds on validation, never on test.

---

## Non-negotiables

1. Access forms (DeeperForensics, KoDF, DeepSpeak) submitted Week 1 — approval
   latency is the one thing extra hours cannot fix.
2. Face caching launches Week 1 WITH landmark sidecars (SBI needs them; re-extraction
   costs a week).
3. Writing starts Week 2, not Week 7.
4. Phase 1 gate is hard: no training until leakage check returns NONE.
5. arXiv on submission day — converts the publication gamble into a guaranteed floor.

---

## Fail-branches

| Gate | If it fails |
|---|---|
| Phase 0 multi-clip | multi-clip WORSE than single → bug in clip extraction, fix before Phase 1 |
| Phase 2 aggregation | no gain → sampling wasn't the bottleneck, reallocate time to Phase 3 |
| Phase 3 SBI | no gain → publishable as "SBI does not transfer to clip-based detectors"; reframe to honest-collapse story |
| Phase 4 temporal | flat ordering → clean negative result, corroborates frame-level-dominance finding |
