# TODO — dataset download tasks (what we're actually going to do)

The **action plan**. Sizes, roles, per-dataset facts, and storage reasoning live in the
registry: **`DATASETS.md`** (same folder). This file is just the tasks + their order + what's
blocking each. **Nothing downloads until the user explicitly says so.**

**Status: NOTHING DOWNLOADED YET.**

---

## What I need from you

1. ~~**Kaggle API token**~~ — **DONE.** Token is in place (`/home/max/.kaggle/access_token`),
   CLI authenticated, `kanzeus/realai-video-dataset` already queryable (size confirmed 2.665 GB).
2. ~~**DeeperForensics-1.0 application form**~~ — **DONE, APPROVED.** Email + Drive download link
   received. Dataset is now accessible; just needs your go-ahead to pull (281 GB — see Task 4).

---

## Task 1 — FaceForensics++ : clean full c23 download from scratch  ← the main task

**Decision (2026-09-05): wipe the existing partial FF++ on disk and re-download all 7 types at
c23 fresh**, so the whole corpus comes from one uniform pull. Total ~30.9 GB. Full per-type size
table + the naming-convention blocker writeup are in `DATASETS.md` → FF++ detail.

**Steps — run ONLY when you say "download", NOT before:**

1. `git status` — confirm nothing uncommitted under `datasets/videos/faceforensics++/` (raw
   videos aren't tracked, but check for stray notes/manifests).
2. **DELETE the existing on-disk FF++ tree** (`fake/` + `real/`, the current 4 types, ~26.15 GB).
   This is the "del old ones" step you authorized *at download time* — re-confirm at that moment.
3. Download all 7 wanted types at c23 into a staging dir:
   ```
   python -m proc.dataset_downloader.faceforensics.download_faceforensics \
       datasets/videos/faceforensics++/_staging_download \
       -d original -d DeepFakeDetection_original -d Deepfakes -d DeepFakeDetection \
       -d Face2Face -d FaceSwap -d NeuralTextures \
       -c c23 -t videos --server EU2 --yes
   ```
   (or `-d all`, then discard the FaceShifter output we don't want).
4. Reorganize staging → `fake/<Manip>/c23/videos/` and `real/{youtube,actors}/c23/videos/`,
   verify counts + sizes against the table in `DATASETS.md`, delete `_staging_download/`.
5. Verify (per-type counts, identity-naming check), then recompute the corpus balance ratios in
   `CONVENTIONS.md` (adding F2F/FS/NT shifts the 25.11%-real figure + the ~2.98 clips/real-video number).

**Run discipline when it actually runs:**
- Background, logged to a file — real multi-GB network-bound pull.
- `--server EU2` primary; fall back to `EU` if EU2 stalls (CA has known issues per repo notes).
- After each type: report count landed, total size, one `ffprobe` check (real h264 c23, not an
  HTML error page saved as `.mp4`).
- If access fails partway: STOP and report the exact error — don't retry blindly or switch
  servers without saying so first.

**Expected final state: ~30.9 GB, 8,395 videos, all c23, all 7 types.**

**Unresolved before running:** naming-convention fix — stage-then-reorganize (recommended,
mechanical) vs. rewriting `DatasetBuilder` FF++ discovery (bigger, deferred). Detail in
`DATASETS.md` → FF++ detail → naming convention mismatch.

---

## Task 2 — real-ai-videos (Kaggle)  ← clean re-download

Confirmed: the Kaggle set (`kanzeus/realai-video-dataset`) IS the existing on-disk
`real-ai-videos/` (same 66 videos). **Decision: re-download it clean from Kaggle** into
`datasets/videos/real-ai-videos/`. Ready to go (CLI authenticated, 2.665 GB).

**Steps (when you say download):** wipe existing `datasets/videos/real-ai-videos/`, then
`kaggle datasets download -d kanzeus/realai-video-dataset` + unzip into place, verify 66 files +
`ffprobe` a sample. Re-check the 4K/no-face quirk. Detail in `DATASETS.md` → real-ai-videos.

---

## Task 3 — DFDC  ← full set FLAGGED NOT FEASIBLE; only a small subset is obtainable

**Full 471 GB = parked, not feasible** (2026-09-07): Kaggle CLI 404s on the chunks (external GCS
data), browser needs session cookies + rules/phone-verification, AWS/dfdc.ai is flaky + maybe
charged. See `DATASETS.md` → "DFDC full set — FLAGGED NOT FEASIBLE".

**Only realistic pulls if a DFDC test slice is still wanted:**
- 4.14 GB official subset via CLI (free): `train_sample_videos` (401 labeled) + `test_videos` (400 unlabeled).
- Unofficial Kaggle mirrors (real data, unofficial hosting): `pranay22077/dfdc-10` (~103 GB, first 10 parts),
  `dsabljic/dfdc-sample` (~60 GB), `aleksandrpikul222/dfdcdfdc` (~11 GB).

Not started. Decide later whether DFDC is worth pursuing at all given the friction.

---

## Task 4 — DeeperForensics-1.0  ← APPROVED, ready on your go-ahead (NEXT UP)

TEST-only. **Access granted** — email + Drive link received:
`https://drive.google.com/drive/folders/1s3KwYyTIXT78VzkRazn9QDPuNh18TWe-`. Only blocked on your go-ahead.

**💡 Only need ~128 GB, not 281 GB:** download the **12 `manipulated_videos_part_*.zip` (fakes,
~127.5 GB) + lists** and **SKIP the 17 `source_videos_*.zip` (~183 GB)** — source videos aren't a
detection class. The real class comes from FF++ youtube originals (Task 1), matched by the 3-digit
target id in `<target_id>_<source_id>.mp4`.

**Steps (when you say download):**
1. Install `gdown` (ask first) or use per-file ranged pulls of the 12 manipulated archives.
2. Download the 12 `manipulated_videos_*` archives + `lists` to `datasets/videos/deeperforensics-1.0/`.
3. Unzip (adapt `unzip.sh`, or manual), verify counts (should total 11,000 fakes across 11 variant folders).
4. Wire real class from FF++ youtube originals; respect the anti-leak rule (group by source identity,
   use official `lists/splits/` train/val/test 703:96:201). Detail in `DATASETS.md` → DeeperForensics-1.0.

---

## Task 5 — DeepSpeak v2.0  ← BLOCKED: HF access not approved yet

TEST — temporal-decay / unseen-generator + audio-visual (Phase 6/7). **124.53 GB** (measured),
self-contained (ships real + fake + splits, no FF++ dependency). Modern 2025 generators (facefusion,
diff2lip, hellomeme, latentsync, liveportrait, memo + voice engines).

**⚠️ Blocked (corrected 2026-09-08):** token is valid but the gated **access request is NOT approved** —
all data files return 403 ("not in the authorized list"). Metadata/README being visible ≠ access.
**You must:** on the HF dataset page (logged in), click "Agree and access repository" / complete the
form, then wait for manual approval. Only then does download work.

**Steps (when you say download):**
```
hf download faridlab/deepspeak_v2 --repo-type dataset --local-dir datasets/videos/deepspeak-v2/
```
Detached, verify the 25 archives + 3 annotation CSVs land, unzip, check real/fake split from
`annotations-*.csv`. Detail in `DATASETS.md` → DeepSpeak v2.0.

## Order

**Everything below is HELD — nothing downloads until you say go (your call, 2026-09-07).**

1. **Task 1 (FF++ clean c23, ~30.9 GB)** — READY; small, feeds DeeperForensics' real class. Only open item: naming-convention approach.
2. **Task 2 (real-ai-videos re-download, 2.665 GB)** — READY, quick.
3. **Task 4 (DeeperForensics fakes, ~128 GB)** — READY, ✅ approved, `gdown` installed, route verified.
4. **Task 5 (DeepSpeak v2.0, 124.53 GB)** — READY, ✅ HF access granted, token saved.
5. **Task 3 (DFDC)** — only the 4.14 GB CLI subset is feasible; full 471 GB flagged not feasible.

Total for the 4 ready full-datasets (1+2+4+5) ≈ **286 GB** to `D:` (1.4 TB free — fits easily).

Not yet investigated: ForgeryNet (Phase 6, later). KoDF — **rejected** (not pursuing).
