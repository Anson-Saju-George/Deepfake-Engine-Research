# Dataset Decisions — Reference

**The one rule everything follows:** train from a SINGLE source (FF++ c23). Every other
dataset is evaluation-only, permanently. This is what frees Celeb-DF to be a genuine
cross-dataset test and keeps your numbers comparable to published tables.

**Second rule:** real and fake must come from the same source videos (source-paired).
Mixing reals from one dataset with fakes from another teaches the model "which dataset
is this," not "is this manipulated."

---

## TRAIN — one dataset only

| Dataset | Composition | Split | Identity key |
|---|---|---|---|
| **FF++ c23** (Deepfakes, Face2Face, FaceSwap, NeuralTextures) | 1,000 real / 4,000 fake | Official 720 / 140 / 140 videos | `000_003.mp4` → `000` |

Why FF++: natively 1:1 per manipulation, source-paired (fakes generated FROM the reals),
and it is the substrate every cross-dataset table in the literature uses.

### Video-level counts

| Split | Real videos | Fake videos | % real |
|---|---|---|---|
| train | 720 | 2,880 | 20% |
| val | 140 | 560 | 20% |
| test | 140 | 560 | 20% |

### Clip-level after asymmetric sampling (4 clips/real, 1 clip/fake)

| Split | Real clips | Fake clips | % real |
|---|---|---|---|
| train | 2,880 | 2,880 | **50%** |
| val | 560 | 560 | **50%** |

Balance the TRAIN split (and val, for model selection). **Do not rebalance test** —
evaluate at natural composition, aggregate clips to one score per video, and report
video-level AUC as the primary metric. AUC is balance-invariant; that is the whole point.

Clip config: **16 frames @ stride 4**, 224×224, multi-clip at inference.

---

## TEST-ONLY — never train on these, ever

| Dataset | Real / Fake | % real | Purpose | Phase |
|---|---|---|---|---|
| **FF++ c23 test** | 140 / 560 | 20% | in-dataset baseline | 2–5 |
| **FF++ LOMO** | held-out manipulation | — | unseen generator | 3 |
| **Celeb-DF v2 official test** | 178 / 340 | 34% | canonical cross-dataset | 3 |
| **DFDC-Preview** | 1,131 / 4,119 | 22% | second canonical cross-dataset | 3 |
| **DeeperForensics-1.0** | ~1,000 unique fakes on FF++ real base | — | unseen manipulation | 3 |
| **DeeperForensics-1.0 perturbed** | ~10k distortion variants, 7 types × 5 levels | — | robustness | 0 / 3 |
| **KoDF** | 62,166 / 175,776 | 26% | demographic + domain shift | 6 (optional) |
| **DeepSpeak v1.1 / v2.0** | 500 identities, 100+ hrs | — | temporal decay | 6 |
| **ForgeryNet** `public_test_videos.tar` only | ~45% real overall | 45% | method diversity | 6 (optional) |

### Leave-one-manipulation-out (LOMO) — free, no downloads

| Config | Train on | Test on |
|---|---|---|
| LOMO_Deepfakes | F2F, FS, NT | Deepfakes |
| LOMO_Face2Face | DF, FS, NT | Face2Face |
| LOMO_FaceSwap | DF, F2F, NT | FaceSwap |
| LOMO_NeuralTextures | DF, F2F, FS | NeuralTextures |

---

## Images (secondary — reference point only)

| Dataset | Status | Note |
|---|---|---|
| AI-vs-Real | KEEP | 59,988 balanced |
| CIFAKE | DROP / de-emphasise | upscaled 32×32 CIFAR, not faces, near-saturated |

The image track stays as a controlled reference. Do not let it become half the paper —
and do not claim a "modality gap" from it without scoping, because your image data is
fully synthetic while your video data is face-manipulated. That is a manipulation-type
difference wearing a modality costume, and ForgeryNet's own results (video-level AUC
97.28 vs image-level 91.02) point the other way.

---

## DROPPED — and why

| Dataset | Reason |
|---|---|
| Real-AI Videos (66 clips) | Too small to support any conclusion; forced the awkward `video_all` vs `video_combined` split |
| UADFV (98 videos) | 2018-era FakeAPP, effectively solved, 49 reals cannot carry a statistic |
| DFDC full (~470 GB) | Marginal gain over DFDC-Preview for enormous cost |
| ForgeryNet training set (~500 GB) | Breaks comparability (nobody trains on it); image and video subsets have different subjects; long-tailed method skew |
| Celeb-DF v2 as TRAIN | Must stay test-only or you lose the canonical cross-dataset benchmark |

---

## Phase-by-phase dataset usage

| Phase | Train | Evaluate on | New downloads |
|---|---|---|---|
| 0 — free wins | none (existing ckpts) | current test splits + perturbations | none |
| 1 — corpus rebuild | — | — | DFDC-P; verify FF++ / Celeb-DF |
| 2 — Tier A aggregation | FF++ c23 | FF++ test | none |
| 3 — Tier B SBI + generalization | FF++ c23 (SBI: FF++ **reals only**) | FF++ test, LOMO, Celeb-DF, DFDC-P, DeeperForensics | DeeperForensics |
| 4 — Tier C temporal | FF++ c23 | FF++ test + cross-dataset | none |
| 5 — statistics | FF++ c23 × 3 seeds | all of the above | none |
| 6 — temporal decay | FF++ c23 | Celeb-DF → DFDC → DeepSpeak v1.1 → v2.0 | DeepSpeak, KoDF |
| 7 — audio-visual | DeepSpeak / FakeAVCeleb | — | FakeAVCeleb |

Note on Phase 3: SBI trains on **real videos only** — it synthesises pseudo-fakes by
blending a face with a perturbed copy of itself. No fake data is used at training time,
which is exactly why it generalises across generators.

---

## Identity conventions (splits depend on these)

| Dataset | Pattern | Identity |
|---|---|---|
| FF++ real | `000.mp4` | `000` |
| FF++ fake | `000_003.mp4` | `000` (target sequence) |
| Celeb-DF real | `id0_0000.mp4` | `id0` |
| Celeb-DF fake | `id0_id16_0004.mp4` | `id0` |
| anything else | — | **must fail loudly**, not silently become its own identity |

---

## Access status — submit these early

| Dataset | Access | When needed |
|---|---|---|
| FF++ c23 | on disk | Phase 1 |
| Celeb-DF v2 | on disk | Phase 3 |
| DFDC-Preview | open download (~20 GB) | Phase 3 |
| DeeperForensics-1.0 | form + edu email | Phase 3 |
| KoDF | application | Phase 6 |
| DeepSpeak | free to academic institutions | Phase 6 |
| ForgeryNet public test | request | Phase 6 |

Approval latency is the one thing extra working hours cannot fix. Submit the forms in
Week 1 even for Phase 6 datasets.
