# papers/ — KICKSTART PROMPT (read this to begin the project)

You are my research collaborator on a computer-vision paper (deepfake / AI-generated-video
detection). Treat this as a **scientific project with a hard deadline**, not a coding task:
the numbers here go into a peer-reviewed paper, and **the previous version was rejected** — a
silent correctness bug is worse than a crash. You have standing authority to disagree, refuse
work that compromises validity, mark things blocked rather than guess, and tell me when I'm
optimising the wrong thing.

You've been told to read the root `PROJECT_CONTEXT.md`, then this file, then everything in
`papers/`. This file is the map + state + rules so you start fully loaded.

---

## 1. What this project is (30-second version)

**Thesis (hypothesis, not assertion — the experiment decides):** face manipulation and fully
synthetic video are *related but non-identical forensic regimes*. We measure **bidirectional
cross-regime transfer**, test whether **balanced joint training** yields a genuinely unified
representation, and **audit** everything against dataset/source shortcuts. The contribution is
that progression — **not** "another universal detector."

**Two regimes:** *face manipulation* (real video, face swapped/reenacted → local blend/warp
artifact; trained on **FF++**) vs *fully synthetic* (whole clip generated → whole-frame
statistics; trained on **GenVideo**).

## 2. Read in THIS order (authoritative → context → evidence)

1. **`writing/01_START_HERE/`** — `PROJECT_VISION.md` (why) → `RESEARCH_QUESTIONS.md` (the 3 RQs = the contribution) → **`EXPERIMENT_DESIGN.md` (AUTHORITATIVE — how; §7 = extraction architecture)** → `HANDOFF_PROMPT.md`.
2. **`writing/02_references/`** — `READING_LIST.md` (tiered; read the 4 ⚡ papers first) + `REFERENCES.bib`.
3. **`writing/03_dataset_intel/`** — `DATASET_DECISIONS.md`, `DATASETS_FINAL.md`, `FFPP_*`, `FORGERYNET_BRIEF.md`.
4. **`literature/`** — 42 PDFs, tiered `tier1_core/` → `tier2_methods/` → `tier3_reference/` (filenames are `NN_Name_author.pdf` in reading order).
5. **`writing/05_provenance_prompts/`** — how the prior benchmarks were produced (provenance, not instructions).
6. **`writing/06_superseded/`** — OLD plan/manuscript. Context only; **do not** reinstate.
7. **`Deepfake-AI-Conversation.pdf`** — full Claude+ChatGPT planning transcript (381 pg) if you need the reasoning trail.

## 3. The three research questions (from `RESEARCH_QUESTIONS.md`)

- **RQ1 — cross-regime transfer & its (a)symmetry.** F→S? S→F? Is `F→S ≈ S→F`? Hypothesis: **not symmetric**.
- **RQ2 — unified training.** Does `FF++ + GenVideo` beat specialists on-domain *and* hold cross-domain — and does it *genuinely unify* the representation or just learn two decision regions? Probe with embeddings / domain probes.
- **RQ3 — audited generalization.** Do RQ1/RQ2 survive shortcut removal? Use VidAudit-style controls + a 32×32 thumbnail diagnostic.

Every RQ is publishable in either outcome. **Run the transfer matrix before anything expensive.**

## 4. Extraction architecture — DECIDED (`EXPERIMENT_DESIGN.md §7-8`)

- **Whole-frame. No face detection, no face cropping.** (Old face-crop pipeline is legacy.)
- **Two layers:** thin **per-dataset adapters** (the only dataset-specific code — walk each dataset's real on-disk wiring; parse filename→identity/label/regime/manipulation/generator; own the real/fake rule and `source_real_dataset` linkage) that emit normalized `VideoRecord`s → a **shared dataset-agnostic core** (pyav decode → **canonical codec + duration normalize** → sample → write frames + metadata). ~90% of logic lives in the core; a new dataset = one thin adapter.
- **Order:** pull dataset → inspect real structure → write its adapter → run the core. Core built once, up front. **Do NOT preprocess 500 GB before the design is validated** on a small slice.
- Metadata schema (per frame) is mandatory — without `dataset`/`regime`/`source_real_dataset` the RQ3 audits can't run.

## 5. Models / metrics (design decisions, don't re-litigate)

- Backbones: **ConvNeXt-B** + a foundation encoder (**SigLIP-So400M** / DINOv2). Temporal-head sweep is **CUT** — the independent variable is *training regime*, not architecture. Frame aggregation = mean-pool.
- Metrics: **audited tuple** — video-level AUROC + above-floor margin + **recall @ 0.1% FPR** + calibration. **Report per-dataset, never a pooled headline.** Threshold frozen on train-domain val. Multi-seed (3) + Wilcoxon + bootstrap CIs.
- Datasets: FF++ (face TRAIN) · GenVideo (synthetic TRAIN, subsample ~10-20k for the controlled comparison) · DeeperForensics/Celeb-DF (face OOD) · DeepAction (synthetic OOD, test-only). ForgeryNet **dropped** from core. DFDC **rejected**. Full registry: `proc/dataset_downloader/DATASETS.md` (+ `TODO.md`).

## 6. Environment (CRITICAL — read `perf/PERFORMANCE_CONTEXT.md`)

- Machine **`OMEN-MAX`**: RTX 5080 Laptop **16.3 GB** (`sm_120`), Core Ultra 7 255HX (20 cores), 63 GB RAM, Windows 11 + **WSL2 Ubuntu**. Invoke WSL from the Windows Bash tool: `wsl.exe -d Ubuntu -e bash -lc "…"`.
- **`/opt/ml`** = lean **production/benchmark** venv (torch 2.11.0+cu129, cuDNN 9.17.1.4) — **NEVER install into or modify it**; benchmark reproducibility depends on it.
- **`/opt/dev`** = full-stack AIML dev venv (`/opt/dev/bin/python`) — do experimental work here (torch cu129 + transformers/anthropic/openai/jupyter/fastapi/etc.). Jupyter kernel `opt-dev`.
- **No passwordless sudo** — anything needing root, hand me the command to run.
- **Winning training knobs (measured, `perf/PERFORMANCE_LOG.md`):** AMP **bf16** · batch **4** · workers **12** · seq_len **16** · **pyav** CPU decode · cache on **F:**. `torch.compile` OFF, `channels_last` OFF.
- **⚠ Silent VRAM cliff:** past ~16.3 GB (batch ≥12, or seq32/batch4) WSL/WDDM spills to RAM — ~16× slowdown, **no OOM error**. Tune with this in mind.

## 7. State from the last session (what's already done)

- Repo cleaned: `test/`+`docs/` deleted (content → `perf/` + `books/`), `train/video/` archived to `temp/legacy_models/`, root tidied. `books/` (01–12 lifecycle) is the paper-writing scaffold but is **OLD image-vs-video framing** — reconcile against `EXPERIMENT_DESIGN.md` first.
- `papers/` restructured into this bundle; 42 papers downloaded+tiered; `Deepfake-AI-Conversation.pdf` assembled.
- Envs `/opt/ml` + `/opt/dev` built & documented (env runbook lives in the **dev-stash** repo: `ENVIRONMENTS.md`, `HP-OMEN-ML-PERF.md`).
- **`extract_frames.py` / the shared core does NOT exist yet** — that's the first real build.

## 8. Codex — your peer collaborator (USE IT)

Codex (OpenAI, `gpt-5.6-sol`) is installed as a **peer agent**, not a tool. Pull it in for
design/architecture/critical decisions and adversarial review — tell it to **fight, not
flatter**; announce every call (each spends usage); the stop-time **review gate stays OFF**
unless I ask. Commands: `/codex:review`, `/codex:adversarial-review` (steerable), `/codex:rescue <task>`, `/codex:status|result|cancel`, `/codex:transfer`, plus the `codex:codex-rescue` subagent.

> **IMMEDIATE NEXT ACTION:** before writing any code, **convene a Codex adversarial review** of
> the **extraction architecture (`EXPERIMENT_DESIGN.md §7`)** and the **three RQs** — poke holes
> in the adapter/core boundary, the metadata schema, the leakage-safe source-real linkage, and
> whether the transfer matrix actually answers RQ1. Then land the design and start building.

## 9. Hard rules & gotchas (do not violate)

- **Git commits: NEVER add a `Co-Authored-By: Claude` / any AI-attribution trailer.** Write as the user.
- **`datasets/` is a git submodule** — its pointer stays unstaged; never commit multi-GB data there. Repo remotes: main repo `Anson-Saju-George/Deepfake-Engine-Research`; env docs `.../dev-stash`.
- **Leakage rules (non-negotiable, `EXPERIMENT_DESIGN.md §1`):** DeeperForensics ships no reals → its real class = FF++ YouTube originals, split by FF++ target id FIRST. Never "fakes from A vs random reals from B" — pair reals to fakes by source/domain; report matched-real and cross-domain-real separately.
- **Duration + codec normalization is a preprocessing rule, not optional** — clip length alone can separate regimes (VidAudit hit 0.998 AUC on nothing but duration).
- When I hand you a file from Downloads to import, **delete the Downloads source after** confirming it's committed (no redundant copies).
- **Label convention:** `real = 1`, `fake = 0`. F1 positive-class was historically inconsistent — always state which class is positive.

## 10. Your first moves in the new chat

1. Confirm you've absorbed §1–§9 (one tight paragraph — don't regurgitate).
2. Propose the **Codex adversarial-review** framing for the extraction architecture + RQs (§8), and run it when I say go.
3. After the review lands: pick the **first dataset to pull** (FF++ clean c23 is the natural start — see `DATASETS.md`/`TODO.md`), write its **adapter**, scaffold the **shared core**, validate on a small slice, then Phase A of the transfer matrix.

Optional deferred cleanup (not blocking research): fill the 33 `[VERIFY]` tags in `REFERENCES.bib`; `temp/legacy_models` 24 GB purge; `books/` trim/whole-frame reconciliation.

**Now begin.**
