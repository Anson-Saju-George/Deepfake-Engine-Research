# Monitoring & Maintenance

## Maintenance Priorities

- keep docs aligned with the active code path
- verify dataloader counts after dataset changes
- track frame extraction progress and storage growth
- monitor for stale manifests or partial frame folders
- keep research notes synchronized with any methodology changes

## Monitoring Questions

- do real filesystem counts still match dataloader discovery?
- have any new corrupt files appeared?
- has preprocessing introduced inconsistent derived artifacts?
- do docs still reflect the active split and training strategy?

## Practical Tools

- `python -m data.dataset_analyzer`
- `python -m data.video_frame_stats`
- `python -m data.image_video_frame`
- `python train_data_pipeline_pull.py`

## Expected Healthy State

- every configured raw dataset reports matching on-disk and loader totals
- image cleanup regressions do not reintroduce unreadable files
- video extraction reruns skip complete frame folders and only refresh stale or partial outputs
- documentation remains consistent with the active CPU preprocessing path
- run directories in `train/image/` remain internally consistent with saved summaries and checkpoint paths

## When to Update the Book

Update the relevant lifecycle file when any of the following changes:

- dataset counts
- split policy
- active preprocessing path
- training protocol defaults
- accepted or rejected experimental branches
