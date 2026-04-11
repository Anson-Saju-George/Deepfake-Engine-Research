"""Video experiment registry for the new train pipeline.

This registry is intentionally high level. It records the planned raw-video
experiment ladder before trainer implementation lands in the new ``train/``
tree.

The video plan is split into three categories:

- spatial: single-frame baselines sampled from raw videos
- temporal: sequence models emphasizing ordered clip dynamics
- spatial_temporal: sequence models that combine strong frame reasoning with
  richer temporal aggregation
"""

from copy import deepcopy


GLOBAL_VIDEO_DEFAULTS = {
    "protocol": "video_only",
    "raw_video_only": True,
    "frame_materialization_required": False,
    "epochs": 10,
    "balanced_sampling": True,
    "split_strategy": {
        "videos": "identity_aware_70_10_20",
    },
    "clip_sampling_train": "random",
    "clip_sampling_eval": "center",
    "target_threshold": 0.95,
    "dataset_scope_options": [
        ["celeb-df-v2"],
        ["faceforensics++"],
        ["celeb-df-v2", "faceforensics++"],
    ],
}


PRIMARY_VIDEO_EXPERIMENTS = {
    "VID-SPA-01": {
        "experiment_no": "VID-SPA-01",
        "category": "spatial",
        "family": "ConvNeXt",
        "model_name": "convnext_base",
        "mode": "single",
        "seq_len": 1,
        "batch_size": 16,
        "reason": (
            "Primary raw-video spatial baseline. Uses one sampled frame per "
            "video and keeps the backbone at the legacy floor already trusted "
            "in the old training tree."
        ),
        "description": (
            "Video-domain spatial baseline for testing manipulation cues "
            "without explicit temporal modeling."
        ),
        "why_now": (
            "Needed before any temporal claim. If this is already strong, "
            "temporal gains must beat a serious frame baseline."
        ),
        "legacy_anchor": [
            "video_celebdf",
            "video_ffpp",
            "video_combined",
        ],
        "status": "planned",
    },
    "VID-SPA-02": {
        "experiment_no": "VID-SPA-02",
        "category": "spatial",
        "family": "Xception",
        "model_name": "xception71",
        "mode": "single",
        "seq_len": 1,
        "batch_size": 16,
        "reason": (
            "Research-facing forensic CNN baseline with a different inductive "
            "bias from ConvNeXt."
        ),
        "description": (
            "Second raw-video spatial baseline used to test whether the "
            "backbone family materially changes frame-level sensitivity."
        ),
        "why_now": (
            "Useful as a strong comparator to the ConvNeXt baseline before "
            "moving into sequence-heavy experiments."
        ),
        "legacy_anchor": [
            "baseline_xception",
            "combined_xception",
        ],
        "status": "planned",
    },
    "VID-TMP-01": {
        "experiment_no": "VID-TMP-01",
        "category": "temporal",
        "family": "ConvNeXt Sequence",
        "model_name": "convnext_base",
        "mode": "sequence",
        "seq_len": 8,
        "batch_size": 8,
        "reason": (
            "Primary temporal baseline using contiguous raw-video clips with a "
            "simple strong backbone."
        ),
        "description": (
            "Sequence model focused on ordered clip dynamics while keeping the "
            "aggregation path simple."
        ),
        "why_now": (
            "This is the cleanest first test of whether temporal structure "
            "adds signal beyond the spatial baselines."
        ),
        "legacy_anchor": [
            "video_celebdf",
            "video_ffpp",
            "video_combined",
        ],
        "status": "planned",
    },
    "VID-TMP-02": {
        "experiment_no": "VID-TMP-02",
        "category": "temporal",
        "family": "ResNet+LSTM",
        "model_name": "resnet50.a1_in1k",
        "mode": "sequence",
        "seq_len": 16,
        "batch_size": 8,
        "reason": (
            "Explicit temporal aggregation experiment using a recurrent head "
            "over ordered frame embeddings."
        ),
        "description": (
            "Longer contiguous-clip temporal model intended to test whether a "
            "dedicated temporal head helps over plain clip averaging."
        ),
        "why_now": (
            "Provides a real temporal-design comparison instead of only "
            "changing the clip length."
        ),
        "legacy_anchor": [
            "resnet_lstm",
        ],
        "status": "planned",
    },
    "VID-ST-01": {
        "experiment_no": "VID-ST-01",
        "category": "spatial_temporal",
        "family": "ConvNeXt Hybrid",
        "model_name": "convnext_base",
        "mode": "sequence",
        "seq_len": 12,
        "batch_size": 8,
        "reason": (
            "Hybrid clip model that tries to preserve strong per-frame "
            "representation quality while adding richer clip-level "
            "aggregation."
        ),
        "description": (
            "Spatial+temporal hybrid experiment inspired by the legacy top-k "
            "and frequency-combined branch."
        ),
        "why_now": (
            "This is the first higher-complexity video experiment after the "
            "plain spatial and plain temporal baselines are established."
        ),
        "legacy_anchor": [
            "topk_convnext",
            "freq_convnext",
            "combined_convnext",
        ],
        "status": "planned",
    },
    "VID-ST-02": {
        "experiment_no": "VID-ST-02",
        "category": "spatial_temporal",
        "family": "Xception Hybrid",
        "model_name": "xception71",
        "mode": "sequence",
        "seq_len": 12,
        "batch_size": 8,
        "reason": (
            "Hybrid forensic-style clip model using xception71 as the frame "
            "backbone with richer clip-level aggregation."
        ),
        "description": (
            "Spatial+temporal hybrid experiment for testing whether a "
            "forensic-style frame encoder benefits from temporal aggregation."
        ),
        "why_now": (
            "Acts as the xception-side counterpart to the ConvNeXt hybrid "
            "experiment."
        ),
        "legacy_anchor": [
            "topk_xception",
            "freq_xception",
            "combined_xception",
        ],
        "status": "planned",
    },
}


CATEGORY_GUIDE = {
    "spatial": {
        "definition": (
            "One sampled frame from each raw video. This is still video-domain "
            "training, but without explicit temporal modeling."
        ),
        "signal_type": "Spatial forensic cues only.",
        "loader_behavior": "dtype=video, mode=single, seq_len=1",
    },
    "temporal": {
        "definition": (
            "Ordered contiguous clips sampled from raw videos. The model is "
            "expected to learn temporal consistency and motion dynamics."
        ),
        "signal_type": "Primarily temporal consistency over ordered frames.",
        "loader_behavior": "dtype=video, mode=sequence, contiguous clips",
    },
    "spatial_temporal": {
        "definition": (
            "Clip models combining strong per-frame spatial reasoning with "
            "richer temporal aggregation or selection."
        ),
        "signal_type": "Joint spatial and temporal evidence.",
        "loader_behavior": "dtype=video, mode=sequence, richer clip modeling",
    },
}


def get_video_experiment_registry():
    """Return the video experiment registry merged with global defaults."""
    registry = {}
    for exp_no, config in PRIMARY_VIDEO_EXPERIMENTS.items():
        merged = deepcopy(GLOBAL_VIDEO_DEFAULTS)
        merged.update(config)
        registry[exp_no] = merged
    return registry


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_video_experiment_registry())
