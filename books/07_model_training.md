# Model Training

## Training Tree

- current transition state:
  - new image training tree is now active under `train/`
  - legacy video references still exist under `train_old/`
  - the new video tree is currently planning/registry only

Current image path components:

- `train/image_model.py`
- `train/image_config.md`
- `train/image_simulate.py`
- `train/image_train_vit.py`
- `train/run_image_vit.py`

Legacy reference components retained for comparison:

- `train_old/image_models/*`
- `train_old/video_models/*`
- `train_old/multi_models/*`

Important interpretation:

- image and video branches remain the intended main experiment families
- the multi branch remains auxiliary rather than the default core protocol
- image training has begun migrating into the new `train/` tree first

## Default Training Policy

### Images

- use `image_only`
- preserve source train/test
- derive validation from source train
- this is the image-domain spatial benchmark family
- current implemented trainer status:
  - smoke test validated on both image datasets
  - ViT experiments `IMG-EXP-01..03` have a working trainer and runner
- current image save layout:
  - `train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/`
- current image record files:
  - `config.json`
  - `history.csv`
  - `best_summary.json`
  - `final_summary.json`
  - `split_summary.json`
  - `run_record.md`

### Videos

- use `video_only`
- use identity-aware `70/10/20`
- use contiguous clip sampling
- split each video dataset independently before combining them
- respect dataset identity grouping where naming conventions support it
- `mode="single"` is the video-domain spatial baseline
- `mode="sequence"` is the video-domain spatial+temporal path

### Frames

- use `frame_only`
- use identity-aware `70/10/20`
- treat as derived-video experiments
- do not treat frame extraction as a required precursor to main video training

## Sampling Policy

- train clip sampling: random contiguous clip
- eval clip sampling: center contiguous clip

Loader mode semantics:

- `mode="single"` for image or frame-level single-sample training
- `mode="sequence"` for contiguous raw-video clips or contiguous frame-folder sequences

Current default sequence length in the loader path:

- `seq_len = 8`

Operational meaning:

- raw videos remain the source of truth
- the model does not ingest the full raw video tensor in one step
- the loader samples one frame for the spatial baseline or a contiguous clip for the spatial+temporal path

## Current Baseline Targets

- CIFAKE-only image
- AI-generated-only image
- combined image
- Celeb-DF-only video spatial baseline
- FaceForensics++-only video spatial baseline
- Celeb-DF-only video
- FaceForensics++-only video
- combined video

## Balancing Policy

- balancing is applied after protocol selection and split preparation
- balancing should not be used as a substitute for correct dataset boundaries
- fake-heavy video datasets can be balanced after identity-aware splitting

## Research Guardrails

- do not treat default pooled image+video training as the main benchmark path
- report frame-only results as derived-video experiments
- keep image and video benchmark families separate in the paper
- keep failed image runs in the experiment record instead of deleting them
- use stable run directories and stable checkpoint names rather than metric-filled folder names
