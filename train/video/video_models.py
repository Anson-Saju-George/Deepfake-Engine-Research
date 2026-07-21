"""Unified video experiment registry for the active train tree.

The active video layout is organized by category package:

- ``train/video/spa`` for spatial single-frame experiments
- ``train/video/tmp`` for temporal clip experiments
- ``train/video/st`` for spatiotemporal and true-video experiments
"""

from copy import deepcopy

from train.video.spa.video_spatial_models import SPATIAL_VIDEO_EXPERIMENTS
from train.video.st.video_spatiotemporal_models import SPATIOTEMPORAL_VIDEO_EXPERIMENTS
from train.video.tmp.video_temporal_models import TEMPORAL_VIDEO_EXPERIMENTS


GLOBAL_VIDEO_DEFAULTS = {
    "protocol": "video_only",
    "raw_video_only": True,
    "frame_materialization_required": False,
    "epochs": 10,
    "balanced_sampling": True,
    "loss_mode": "none",
    "split_strategy": {
        "videos": "identity_aware_70_10_20",
    },
    "clip_sampling_train": "random",
    "clip_sampling_eval": "center",
    "target_threshold": 0.95,
    "dataset_scope_options": {
        "celebdf": ["celeb-df-v2"],
        "ffpp": ["faceforensics++"],
        "video_combined": ["celeb-df-v2", "faceforensics++"],
        "real_ai_videos": ["real-ai-videos"],
        "video_all": ["celeb-df-v2", "faceforensics++", "real-ai-videos"],
    },
}


PRIMARY_VIDEO_EXPERIMENTS = {}
PRIMARY_VIDEO_EXPERIMENTS.update(SPATIAL_VIDEO_EXPERIMENTS)
PRIMARY_VIDEO_EXPERIMENTS.update(TEMPORAL_VIDEO_EXPERIMENTS)
PRIMARY_VIDEO_EXPERIMENTS.update(SPATIOTEMPORAL_VIDEO_EXPERIMENTS)


CATEGORY_GUIDE = {
    "spatial": {
        "definition": "Single sampled frame from each raw video.",
        "signal_type": "Spatial forensic cues only.",
        "loader_behavior": "dtype=video, mode=single, seq_len=1",
        "save_root": "train/video/spa",
    },
    "temporal": {
        "definition": "Ordered contiguous clips sampled from raw videos.",
        "signal_type": "Temporal consistency and ordered clip structure.",
        "loader_behavior": "dtype=video, mode=sequence, contiguous clips",
        "save_root": "train/video/tmp",
    },
    "spatiotemporal": {
        "definition": "Clip-based fusion including native video transformers.",
        "signal_type": "Joint spatial and temporal evidence.",
        "loader_behavior": "dtype=video, mode=sequence, richer clip modeling",
        "save_root": "train/video/st",
    },
}


def get_video_experiment_registry():
    """Return the merged video experiment registry."""
    registry = {}
    for exp_no, config in PRIMARY_VIDEO_EXPERIMENTS.items():
        merged = deepcopy(GLOBAL_VIDEO_DEFAULTS)
        merged.update(config)
        registry[exp_no] = merged
    return registry
