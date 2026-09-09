# DATASETS — registry (every dataset we've considered)

Reference catalog: roles, on-disk state, **measured** sizes, access, storage. The **action plan
(what/when to download)** lives in `TODO.md` (same folder). Also cross-linked from
`docs/RESEARCH_PLAN.md` (roles) and `docs/REFERENCES.bib` (citations).

**Nothing downloaded yet.** Storage: everything lives on **`D:`** (1.9 TB, 1.4 TB free — extended
Sep 2026; download + train there directly, no other drive to compare). The one exception is the
celeb-df-v2 face-crop *cache*, which already lives on `F:` per `CACHE_LOCATION.md` and stays there.

---

## Master table

| Dataset | Role | On disk | To download | Access | Location (D:) |
|---|---|---|--:|---|---|
| **FaceForensics++** (c23, 7 types) | TRAIN | 4/7 types, ~26 GB (to be wiped) | **~30.9 GB** clean re-pull | ✅ script works | `datasets/videos/faceforensics++/` |
| **Celeb-DF v2** (official test) | TEST (perm) | ✅ complete, 6,533 vids | — | request form `[inferred]` | `datasets/videos/celeb-df-v2/` |
| **real-ai-videos** (Kaggle = the on-disk set) | undeclared | ✅ 66 vids | 2.665 GB (**clean re-download**) | ✅ CLI authed | `datasets/videos/real-ai-videos/` |
| ~~**DFDC**~~ | ❌ **REJECTED** (kept as note) | ❌ | full 471 GB not downloadable here | ❌ no feasible route | — |
| **DeeperForensics-1.0** | TEST (unseen/robustness) | ❌ | **~128 GB** needed (fakes only; full is 281 GB) | ✅ **APPROVED** — email + Drive link received | `…/deeperforensics-1.0/` (proposed) |
| **DeepSpeak v2.0** | TEST (Phase 6/7, temporal-decay + audio-visual) | ❌ | **124.53 GB** measured (self-contained: real + fake) | ⚠️ HF gated — token valid but **access NOT yet approved** (data files 403) | `…/deepspeak-v2/` (proposed) |
| **DeepAction v1** | TEST ONLY (cross-fake-type: synthetic video) | ⏳ downloading | **5.64 GB** (2,605 videos) | ✅ ungated (HF public) | `datasets/videos/deepaction-v1/` (D:) |
| **ForgeryNet** | UNDECIDED (candidate TEST, video) | ❌ | **496.23 GB** (recorded, not fetched) | 🔒 OpenDataLab acct / GDrive; non-commercial | `…/forgerynet/` (proposed) |
| ~~**KoDF**~~ | ❌ **REJECTED** | — | not pursuing | — | — | — |

**Queued to download now: ~289 GB** (FF++ 30.9 clean + real-ai-videos-Kaggle 2.665 if not
redundant + DeeperForensics 281.39). Fits easily in 1.4 TB free.

---

## FaceForensics++ — full size matrix

All 7 wanted types × 3 compression levels. **c23 is what we use** (raw/c40 for reference only).
**c23 column is exact** — byte-summed on disk (4 present types) + ranged-GET on every file (3
missing). c40 exact for the 3 missing, sampled for the rest. raw is sampled projection.

| Type | Part of | Videos | raw | **c23** | c40 |
|---|---|--:|--:|--:|--:|
| `original` (youtube reals) | FF++ | 1,000 | ~82.2 GB | **1.804 GB** | ~0.25 GB |
| `Deepfakes` | FF++ | 1,000 | ~86.5 GB | **1.856 GB** | ~0.25 GB |
| `Face2Face` | FF++ | 1,000 | ~100.4 GB | **1.815 GB** | 0.242 GB |
| `FaceSwap` | FF++ | 1,000 | ~69.7 GB | **1.526 GB** | 0.200 GB |
| `NeuralTextures` | FF++ | 1,000 | ~102.6 GB | **1.419 GB** | 0.191 GB |
| `DeepFakeDetection` (DFD) | DFD | 3,066 | ~897.4 GB | **19.782 GB** | ~1.53 GB |
| `DeepFakeDetection_original` (DFD reals) | DFD | 363 | ~264.9 GB | **2.703 GB** | ~0.45 GB |
| **TOTAL** | | **8,395** | **~1,604 GB** | **~30.9 GB** | **~3.1 GB** |

- **On disk now:** 4 types (`original`, `DeepFakeDetection_original`, `Deepfakes`,
  `DeepFakeDetection`) = ~26.15 GB. Missing: `Face2Face`, `FaceSwap`, `NeuralTextures` (4.76 GB).
- **Plan:** wipe all and re-pull all 7 clean at c23 (see `TODO.md` Task 1).
- **Why c23 is small:** H.264 CRF-23 on short clips → genuinely ~1.5–1.9 MB/video (confirmed on
  real files, not projected). c40 (CRF-40) ~0.2 MB/video. raw is absurd (~1.6 **TB**) — never used.
- `FaceShifter` excluded (not one of FF++'s 4 core manipulations).

**⚠ Integration blocker (naming convention):** the download script writes
`manipulated_sequences/<M>/…` & `original_sequences/{youtube,actors}/…`, but `data/dataloader.py`
requires the literal `fake/`/`real/` path segments. **Fix:** download to a staging dir, then move
into `fake/<M>/c23/videos/` + `real/{youtube,actors}/c23/videos/` (mechanical, no code change).
Alternative (deferred): rewrite `DatasetBuilder` FF++ discovery to be layout-flexible.

---

## Celeb-DF v2

- **Role:** TEST only, permanently. **Complete on disk** (6,533 videos) — nothing to download.
- Face-crop cache already built on `F:` (`F:/deepfake-video-ds/`, 30.36 GB, per `CACHE_LOCATION.md`).
- If more were ever needed: manual request-access form (not scriptable). `[inferred, not repo-verified]`

## real-ai-videos

**The Kaggle set (`kanzeus/realai-video-dataset`) IS the on-disk set** (confirmed by user) — same
66 videos, 2.665 GB (Kaggle) vs 2.647 GB (on disk). Origin now known = this Kaggle dataset.
**Decision: re-download it clean from Kaggle** (same clean-pull approach as FF++), into the
existing `datasets/videos/real-ai-videos/`. Role still undeclared in the formal roles table.
**Known quirk:** some clips are 4K with no clear/detectable face (from celeb-df-v2 extraction
findings) — hurts face-crop yield; re-check after re-download.

## ❌ DFDC — REJECTED (kept as a note; not available to download here, 2026-09-08)

**Composition (per papers — not verifiable from disk, since we can't download it):** full DFDC ≈
**128,154 videos** (~104,500 fake / ~23,654 real); Preview ≈ **5,214 videos** (~4,464 fake / ~750 real).
Kaggle-CLI-accessible subset only: `train_sample_videos` 401 (labeled) + `test_videos` 400 (unlabeled).

## ⚠️ DFDC full set (471 GB) — why it's rejected (2026-09-07/08)

The full 471 GB DFDC has **no automatable download route** from here:
- **Kaggle CLI** → `404` (proven twice) — the 50 train chunks are external GCS data the API doesn't serve.
- **Kaggle browser** → works only with logged-in session cookies, and the account is stuck on the
  rules redirect (needs rules acceptance + likely phone verification first).
- **AWS/dfdc.ai** → account ID ready (`673478370581`) but dfdc.ai is known-flaky ("Invalid AWS
  account ID" errors reported) and may be Requester-Pays (~$42 egress).

**Decision: the full 471 GB is parked as not-feasible for now.** If a DFDC test slice is still wanted,
the realistic options are the 4.14 GB official subset (CLI, free) or an unofficial mirror (see below).
Detail retained below for reference.

## DFDC (Meta / Facebook) — three distinct things, don't conflate

Key for cross-dataset generalization (#2 reviewer objection). Role: TEST only, permanently.
Cite `DFDC2019Preview` (arXiv 1910.08854) for Preview, `DFDC2020` (arXiv 2006.07397) for full.

| Variant | Videos | Size | Labels? | How to get it | Charges? |
|---|--:|--:|:--:|---|:--:|
| **Full train set** | 124k | **471.84 GB** | yes | external GCS (website, accept rules) or AWS S3 — **NOT exposed to Kaggle CLI** | AWS route may be Requester-Pays |
| **Preview** | 5,214 | ~5 GB (unverified) | yes | AWS/IAM via `dfdc.ai` (S3) — no standalone Kaggle dataset exists | possible Requester-Pays |
| **Competition public files** (via Kaggle CLI) | 801 | **4.14 GB** | **partial** | `kaggle competitions` — checked, works, authed | **FREE, no AWS** |

**What the Kaggle CLI actually exposes** (verified Sep 2026, rules already accepted on this account):
- `train_sample_videos/` — **401 videos, 1.74 GB, LABELED** (`metadata.json`, real/fake). ✅
- `test_videos/` — 400 videos, 2.40 GB, **UNLABELED** (competition's hidden test; labels never released).
- `sample_submission.csv`. **The 471 GB train chunks are NOT downloadable via the CLI** (GCS/AWS only).

**Zero-charge options for a DFDC test set (avoid AWS entirely):**
1. **Kaggle `train_sample_videos`** — 401 labeled videos, 1.74 GB, free, immediate. Small but real
   cross-dataset test. Command: `kaggle competitions download deepfake-detection-challenge -f train_sample_videos.zip`.
2. **Preview (5,214 labeled)** — more data, but needs an AWS account + likely small Requester-Pays
   egress charges (~$0.09/GB). Not zero-cost.
3. **Avoid:** the 471 GB full set (`aws s3 cp … --request-payer requester` bills your account).
   Standing rule: do **not** run `kaggle competitions download -c deepfake-detection-challenge` (bulk).

Third-party Kaggle mirrors exist (`dsabljic/dfdc-sample` 60 GB, `pranay22077/dfdc-10` 103 GB,
`aleksandrpikul222/dfdcdfdc` 11 GB) — unofficial, label/quality integrity unverified, use with caution.

**Full-set chunk download URLs (recorded from the Kaggle Data page, competition id `16880`):**
- Full (all chunks in one): `https://www.kaggle.com/c/16880/datadownload/dfdc_train_all.zip`
- First chunk: `https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_00.zip`
- Last chunk: `https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_49.zip`
- Pattern: `…/dfdc_train_part_NN.zip` for `NN` = `00`…`49` (50 chunks, ~9–11.5 GB each, ~471.84 GB total).

**Probed Sep 2026 — these chunks are NOT downloadable via the Kaggle API token:**
- `kaggle competitions download -f dfdc_train_part_00.zip` → **404** (`DownloadDataFile` doesn't know
  the file — it's *external* data hosted off-Kaggle on GCS, not a normal competition file).
- The web endpoint `www.kaggle.com/c/16880/datadownload/dfdc_train_part_00.zip`, requested with
  Range 0-0 both anonymously and with `Authorization: Bearer <api-token>`, returns
  **`text/html` from www.kaggle.com** (a login/rules redirect), not the zip. The API token does not
  authenticate the web/GCS route.

**Conclusion:** the 471 GB train chunks require a **logged-in browser session** (rules accepted), not
the API key. To pull them programmatically we'd need one of:
1. Download directly in the browser (click each chunk while logged in).
2. The **resolved GCS signed URL** after the browser redirect (starts `storage.googleapis.com/…?
   X-Goog-Signature=…`) — usable by `curl`/`wget` with no further auth, but time-limited and
   credential-bearing (don't commit it anywhere).
3. Exported Kaggle **session cookies** (`cookies.txt`) → `curl --cookie` (sensitive; handle like a secret).

Downloading via Kaggle itself is free (no AWS Requester-Pays); the blocker is purely auth, not billing.
Train/test role for DFDC still undecided.

## DeeperForensics-1.0

- **Role:** TEST only (unseen manipulation + robustness). Cite: `jiang2020deeperforensics10`.
- **Size: 281.39 GB** — measured directly off the official Google Drive folder (32 files, ranged
  HTTP), cross-checked vs the README's ~284 GB (agree within ~1 %). 60,000 videos (48,475 source
  1080p + 11,000 manipulated across 11 types; 1,000 target videos shared with FF++, not separate).

  | Component | Size | Files | Need it? |
  |---|--:|--:|---|
  | `source_videos_part_00..16.zip` | ~183.4 GB | 17 | ❌ **NO** — raw actor footage to *build* fakes, not a detection class |
  | `manipulated_videos_part_00..11.zip` | ~127.5 GB | 12 | ✅ **YES** — these are the fake videos |
  | lists / terms / unzip.sh | <1 MB | 3 | ✅ yes (splits + distortion meta) |
  | **Full total** | **281.39 GB** | **32** | |
  | **What we actually need** | **~127.5 GB** | **12 + lists** | manipulated only |

  **💡 Download optimization: skip the 17 `source_videos_*` archives (~183 GB).** Downloading only
  the 12 `manipulated_videos_*` archives + lists cuts this from **281 GB → ~128 GB**.

  **Are source videos "real"? Verified against the dataset's own lists (Sep 2026):** the official
  `splits/{train,val,test}.txt` (703/96/201) list **manipulated** videos (`<target_id>_<source_id>.mp4`),
  never source videos. `source_videos_list.txt` = 48,475 raw actor clips
  (`source_videos/M004/BlendShape/camera_down/…`, i.e. 100 actors × lights × emotions × cameras).
  They ARE genuine unmanipulated footage, BUT the authors explicitly forbid using them as the real
  class, and correctly so: (1) **domain/shortcut confound** — studio rig lighting would let a detector
  cheat "studio=real, YouTube=fake"; (2) **identity leakage** — the fakes are built from these exact
  source faces. The real class MUST be **FF++'s 1,000 YouTube target videos** (our FF++ download
  supplies them). So skip source videos: real footage, but methodologically unusable as the real class.
  (`lists.zip` already fetched + extracted to `/mnt/d/_ddf_meta/lists_extracted/` — 445 KB, has all
  video names + splits + distortion meta; gdown download route verified working.)

- **Access: ✅ APPROVED** — approval email received Sep 2026 (from `deeperforensics@gmail.com`).
  Two routes given: **(1) Google Drive** `https://drive.google.com/drive/folders/1s3KwYyTIXT78VzkRazn9QDPuNh18TWe-`
  → `…/folders/1M2mPKdjtP7UJtDZxrTTW5vG4SK61TQuk` (32 files); **(2) Baidu Wangpan** backup
  `https://pan.baidu.com/s/1aJSNsZ1kb8zlpc3nIEAXBQ` (extraction code in the approval email).
  Files are archives + a `bash unzip.sh` (bash ≥4; review before running). **`gdown` installed (6.2.0)
  and the Drive route is verified working** (fetched `lists.zip` cleanly). The email explicitly
  recommends gdown/rclone/gdrive for terminal download.
- **⚠ 281 GB = bigger than the entire rest of the corpus combined.** Never auto-download —
  needs explicit go-ahead even though access is now granted.

**⚠️ CRITICAL — the "real" videos are NOT in this dataset.** DeeperForensics ships only *fake*
(manipulated) videos + *source* videos. The **source videos (48,475) are NOT the real class** —
they're raw actor footage used to *build* the fakes, not for detector training. The real/"target"
class = **1,000 refined YouTube videos from FaceForensics++ (c23)** — which the non-distribution
agreement means DeeperForensics does NOT include; **we must supply them from our own FF++ download**
(the `original`/youtube set — Task 1 already pulls these). So for a real-vs-fake test set:
DeeperForensics manipulated videos = fake, FF++ youtube originals = real, matched by the 3-digit
target ID in the filename.

**Manipulated-video structure (11 variant folders, 11,000 videos, 1,000 each):**

| Folder | Videos | What it is |
|---|--:|---|
| `end_to_end` | 1,000 | raw DF-VAE face swaps (the base fakes) |
| `reenact_postprocess` | 1,000 | raw, alt post-processing (color/warp/affine) |
| `end_to_end_level_1..5` | 5,000 | base fakes + random distortion at 5 intensity levels |
| `end_to_end_random_level` | 1,000 | base fakes + random-type, random-level distortion |
| `end_to_end_mix_2/3/4_distortions` | 3,000 | base fakes + 2/3/4 **mixed** distortions each |

- **Naming:** manipulated `<target_id>_<source_id>.mp4` (target = 3-digit FF++ id, source = actor id;
  swaps are same-gender). Source `<ID>_light_<dir>_<emotion>_camera_<dir>.mp4`.
- **Perturbations:** 7 types × 5 levels — this is the whole point for us (**robustness testing**:
  how detection degrades under real-world distortions). Distortion meta files in `lists/`.
- **Official splits** (`lists/splits/` train/val/test): video ratio **703 : 96 : 201**, source-identity
  ratio 71 : 9 : 20. A "hidden test set" is NOT released.
- **Anti-leak rule (must follow):** group manipulated videos by **source identity**; keep identities
  unrepeated across splits. For face crops, the authors recommend cropping the *real* (target) videos
  first, recording coordinates, then cropping the matching manipulated videos to the same box.

## DeepSpeak v2.0 (Barrington, Böhacek & Farid, UC Berkeley)

- **Role:** TEST — Phase 6/7 candidate. Modern (2025) generators → **temporal-decay / unseen-generator**
  test; audio-visual → entry point for the **Phase-7 audio-visual** chapter. Cite arXiv 2408.05366.
- **Size: 124.53 GB** (measured Sep 2026 via authenticated HfApi — 25 `.zip` archives, 4–6 GB each,
  sum 124.50 GB; page's "134 GB" was approximate). 52+ hours, `1K–10K` samples, real + fake.
  **Self-contained** — ships its own real class + labels + split (unlike DeeperForensics, no FF++ dependency).
- **Structure (HF `faridlab/deepspeak_v2`):** 25 archives `deepspeak-v2-0.zip … deepspeak-v2-24.zip`,
  + `annotations-real.csv`, `annotations-fake.csv`, `annotations-split-def.csv`, loader `deepspeak_v2.py`.
- **Composition — audio-visual (face + voice both manipulated):**
  - Video engines (6): `facefusion`, `diff2lip`, `hellomeme`, `latentsync`, `liveportrait`, `memo`
    — lip-sync, face-swap, avatar.
  - Voice: `elevenlabs`, `playht`, `speechify`, or real audio.
- **Access: ⚠️ NOT YET APPROVED (corrected 2026-09-08).** Gated `manual`. HF token is valid + persisted
  (`whoami` = Anson-Saju-George), and repo **metadata + README are readable** — but that does NOT mean
  access is granted. **Verified: all data files (`annotations-*.csv`, the 25 zips) return 403
  GatedRepoError — "you are not in the authorized list."** An earlier note here said "GRANTED"; that was
  a misread of metadata visibility. **To unlock:** on the dataset page (logged in), click "Agree and
  access repository" / complete the form, then wait for manual approval (email, like DeeperForensics).
  Only after approval will `hf download faridlab/deepspeak_v2 --repo-type dataset` work.
  License `deepspeak-v2-0-license`: **free for academic institutions**, non-academic may incur fees.
- **Real/fake split: UNKNOWN** — the `annotations-real.csv` / `annotations-fake.csv` that carry the exact
  counts are gated (403), can't be read until access is approved.
- Note: **v1** (also arXiv 2408.05366) is the older release noted in `docs/DATASET_NOTES.md`; this is **v2.0**.

## DeepAction v1 (Böhacek & Farid — arXiv 2412.00526, IJCAI-W 2025)

UC Berkeley / Stanford / Google (done during first author's Google internship). HF:
`faridlab/deepaction_v1` (**single version — no v2 exists**). License: AI videos CC BY 4.0,
real videos Pexels license; academic use.

- **Size: 5.64 GB** (HF-API measured; listing says 6.06 GB). **2,605 videos**, ungated (public).
- **Content = text-to-video HUMAN ACTIONS (walking, running, cooking, dancing) — NOT face manipulation.**
- **7 top-level folders** = 6 T2V generators + 1 real:
  `BDAnimateDiffLightning`, `CogVideoX5B`, `RunwayML`, `StableDiffusion`, `Veo` (pre-release),
  `VideoPoet` + `Pexels` (real: 100 videos, 28 min, 44,475 frames). Each folder has 100 subfolders
  = 100 human-action classes; all clips in a subfolder share one (ChatGPT-generated) prompt.
  `captions.csv` included.
- **⚠️ All videos normalized to 512×512** — every generator, every clip.
- Authors' baseline: fine-tuned CLIP + RBF = 99.1% real-vs-AI accuracy.

**Role: TEST ONLY — never train on it.** Purpose = cross-**fake-type** generalization: train on
FF++ face manipulations, test here on fully-synthetic video. **Expected outcome is FAILURE, and
that's the useful result** — evidence that face-manipulation detectors don't transfer to
fully-synthetic generation.

**⚠️ CONFOUND TO GUARD:** all DeepAction clips are exactly 512×512; FF++/Celeb-DF are varied
resolutions. A model can separate them on **resolution alone** ("512×512 = fake") instead of
detecting generation. So: (1) never put DeepAction in a training mix; (2) normalize test
preprocessing identically across all test sets; (3) a suspiciously **high** score here is a red
flag (shortcut), not a success — a **low** score is the expected, meaningful result.

**Download (authorized):** `snapshot_download('faridlab/deepaction_v1')` → `datasets/videos/deepaction-v1/`
(D:, raw-video location, NOT the crop cache). Prefer direct file download over `load_dataset` (the
loader needs `datasets` 3.0.1–3.0.6 and `/opt/ml` is frozen — don't churn it). Verify after: 7
folders, ~2,600 videos, 512×512 via ffprobe on a sample, actual size vs 6.06 GB listed.

## ForgeryNet (He et al., CVPR 2021 **Oral**) — RECORDED, not downloaded

Authors: Yinan He, Bei Gan, Siyu Chen, Yichun Zhou, Guojun Yin, Luchuan Song, Lu Sheng, Jing Shao,
Ziwei Liu. Repo: https://github.com/yinanhe/ForgeryNet · Project:
https://yinanhe.github.io/projects/forgerynet.html · License: **non-commercial research/education only**.
**Role: UNDECIDED** — candidate TEST set (video). See Concerns before assigning.

**Scale** (authors' comparison table): video clips 99,630 real / 121,617 fake; still images
1,438,201 real / 1,457,861 fake → **2.9M images + 221,247 videos**. 15 manipulation approaches
(7 image-level + 8 video-level), 5,400+ subjects, 36 unique + mixed perturbations (only
DeeperForensics also has mixed), 9,393,574 annotations (6.3M classification labels, 2.9M
manipulated-area masks, 221,247 temporal forgery-segment labels). Largest public face-forgery
dataset by scale/manipulations/subjects/perturbations.

**Confirmed total size: 496.23 GB** (public share listing 2021-08-25; official pages publish no GB
number). OpenDataLab lists 499.5 GB — ~0.8% gap is GB-vs-GiB/label-archive counting, NOT a
different version. **MD5s, not byte count, determine the release.**

**Four tasks** (why packages split): (1) Image Forgery Classification (2/3/n-way over 15
approaches); (2) Spatial Forgery Localization (masks); (3) **Video Forgery Classification** —
**manipulated frames at RANDOM positions by design** (a fixed centre-clip sampler would miss them);
(4) Temporal Forgery Localization.

**⚠️ TWO distributions in circulation — MD5-verify which you get** (label files must match the data
release or annotations misalign):
- *Set A (project page):* `Training.tar` `272d385f3cac5dd6feabc25711c18a41` · `Validation.tar`
  `df320faca2128e022764f8d0e4bc75a1` · `public_test_images.tar` `e4218faa4b934345977202edfbf89301`
  · `public_test_videos.tar` `92b870009cb03832b2c2795b6a35629f`
- *Set B (README, `_with_real` naming):* `Training.tar` `750e34521e61f81486a678d5b76a5ef3` ·
  `Validation_with_real.tar` `c6fd18252b5e0ae04a5b163297b5cb28` · `Test_with_real.tar`
  `7d2c0cf50bdc77890ace1ba4461e1893` · `test_list.tar` `8558b99366ccb93cd9da56f600f0901f` ·
  `train_list.tar` `8bda9b0a6dc7721e5f1d0b6b3793ce2a` · `val_list.tar` `a280bb501166a01b8d376df2b5b9ae13`

**Directory layout** (README): `Train/{images/ (15 fake + 4 real subdirs), videos/ (8 fake + 4 real
subdirs), spatial_localize/ (masks)}`.
- Image list: `<image_name> <binary_cls> <triple_cls> <16cls>`
- Video list: `<seg_length> <video_name> [<start> <count> <fake/real>]*N <binary_cls> <triple_cls> <16cls>`
  (per-segment temporal annotations; for binary use just `<binary_cls>`).

**Manipulation taxonomy:** (a) ID-remained: Talking-head Video, FirstOrder Motion, ATVG-Net, MaskGAN,
SC-FEGAN, StarGAN2, StyleGAN2, DiscoFaceGAN. (b) ID-replaced: BlendFace, MM Replacement, DeepFakes,
FSGAN, FaceShifter, SBS, DSS.

**Package inventory (know all, TAKE only the 7 below):**

| Project-page button | Package | Take? |
|---|---|---|
| DOWNLOAD TRAINING SET | `Training.tar.*` (split) | ❌ inside TRAIN&VAL |
| DOWNLOAD VALIDATION SET | `Validation.tar` | ❌ inside TRAIN&VAL |
| **DOWNLOAD TRAIN&VAL** | Training+Validation combined | ✅ TAKE |
| DOWNLOAD IMAGE TEST SET | `public_test_images.tar` | ✅ TAKE |
| DOWNLOAD VIDEO TEST SET | `public_test_videos.tar` | ✅ TAKE |
| DOWNLOAD TRAINING LIST | `train_list.tar` | ✅ TAKE |
| DOWNLOAD VALIDATION LIST | `val_list.tar` | ✅ TAKE |
| (MD5 set B only) | `test_list.tar` | ✅ TAKE if present |
| README | docs | ✅ TAKE |

**⚠️ DUPLICATION TRAP:** `TRAIN&VAL` is an ALTERNATIVE to separate `Training`+`Validation` — taking
both downloads the same data twice (hundreds of GB wasted). **Use TRAIN&VAL only.** Train/val each
already contain both `images/` and `videos/`; the **test split** is the only place image/video ship
as separate archives (not duplication — same split cut by task, both required). Training ships as
split parts: `cat Training.tar.* >> Training.tar`.

**✅ EXACT DOWNLOAD LIST (this and nothing else):** (1) TRAIN&VAL combined, (2) `public_test_images.tar`,
(3) `public_test_videos.tar`, (4) `train_list.tar`, (5) `val_list.tar`, (6) `test_list.tar` (if
served), (7) README. If the served release uses `Validation_with_real.tar`/`Test_with_real.tar`
naming, map by content + MD5, record which set received.

**Download routes (ranked):**
1. ⭐ **OpenDataLab (preferred)** — https://opendatalab.com/OpenDataLab/ForgeryNet (499.5 GB). Has a
   **CLI/SDK** → resumable + scriptable (decisive for 500 GB; a browser pull will fail partway).
   Needs an OpenDataLab account; check the "CLI/SDK download" tab for the exact command.
2. **Official project page** (Google Drive buttons) — https://yinanhe.github.io/projects/forgerynet.html.
   Authoritative but GDrive at 500 GB is fragile (quota, no resume); split parts need reassembly.
3. ⚠️ **Third-party re-shares (last resort, MD5-verify everything)** — bare GDrive IDs
   (`1conYQXWguAwJ1eEwewHyMBGUtgjgR_sM`, `1CSJOkDR_jJvq7qUGP8oGcwpoPdKGzfgb`,
   `1ZqPqmYdqmq4_HLZu1c5ySXhcR1WQWQ9L`, `14Rqq4C4oHK6FEqyTIiVcQy7wQuMHGQzu`); 115cdn share
   `https://115cdn.com/s/swnk84d3wl3` (password `cvpr`, lists 495.23 GB @ 2021-08-25).

**Known infra problems:** train/val moved to Aliyun Drive (Chinese UI, account required; repo issue
#11 reports a researcher unable to download). ~12+ open issues since 2023, several unanswered (latest
#52, Nov 2025). Expect no maintainer support.

**Concerns (decide the paper role):** (1) **Comparability** — almost nobody trains on ForgeryNet, so
cross-dataset numbers trained on it can't be placed against published FF++-trained tables. (2) Image
and video subsets have **different subjects** — cuts against a protocol-separated image-vs-video
comparison (this project's thesis). (3) **Long-tailed** manipulation distribution in train → new
imbalance problem. (4) 496 GB + days of frame extraction; images an extra large cost. (5) Video task
places manipulated frames at random positions — fixed centre-clip sampling would systematically miss them.

**User's position:** ~1.2 TB free, accepts the 496 GB cost; wants BOTH image and video sets so the
image track can be explored later. Paper role (train vs test, video-only vs image+video) UNDECIDED.

**Action / suggested order when go-ahead comes:** validate the chain first — pull README + the three
`*_list.tar` + `public_test_videos.tar`, MD5-verify, confirm which release set + that labels parse and
layout matches README; ONLY THEN start the large TRAIN&VAL pull (catches wrong-release/label-mismatch
in an hour, not after a multi-day download). Go-ahead must specify: (a) confirm 7-item list, (b) route
(default OpenDataLab CLI), (c) target drive. **Download nothing yet.**

## ~~KoDF~~ — REJECTED

Rejected (user decision, 2026-09-07). Not pursuing — no download, no scoping.

---

## How the sizes were measured (provenance)

| Method | Used for |
|---|---|
| `du` / byte-sum on real files | FF++ 4 present types (c23), on-disk real-ai-videos |
| Ranged-GET every file (`Content-Range`, no body) | FF++ 3 missing manipulations (c23 + c40) — `test/_cache/ffpp_exact_size.py` |
| Kaggle API `dataset_list_files` (summed) | real-ai-videos Kaggle |
| Ranged HTTP over public Drive folder | DeeperForensics-1.0 |
| Small-sample HEAD + extrapolate | FF++ raw column, FF++ c40 for the 4 present types (approximate) |
| Not yet measured | DFDC-Preview total GB, KoDF/DeepSpeak/ForgeryNet |

Superseded: an earlier "24.91 GB FF++ c23" figure came from a 5-file-per-dataset sample that
under-counted `DeepFakeDetection` by ~6 GB; replaced by the exact ~30.9 GB above.
