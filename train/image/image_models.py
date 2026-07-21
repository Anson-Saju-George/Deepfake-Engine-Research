"""Image experiment registry for the active image training pipeline.

This file keeps the primary image-only experiment ladder explicit. The chosen
families and sizes are intentionally above the legacy lower bound already used
in the repo.

Implemented status:
- smoke test exists in ``train.image.simulate_image_train``
- ViT trainer/runner currently exist for ``IMG-EXP-01..03``
- ConvNeXt trainer/runner currently exist for ``IMG-EXP-04..06``
- Swin trainer/runner currently exist for ``IMG-EXP-07..08``
- DeiT trainer/runner currently exist for ``IMG-EXP-09``
- ConvNeXtV2 trainer/runner currently exist for ``IMG-EXP-10..11``
- MaxViT trainer/runner currently exist for ``IMG-EXP-12``
- EVA trainer/runner currently exist for ``IMG-EXP-13..14``
- save layout uses ``train/image/<family_name>/<run_name>/``
"""

from copy import deepcopy


GLOBAL_IMAGE_DEFAULTS = {
    "protocol": "image_only",
    "epochs": 10,
    "batch_size": 64,
    "balanced_sampling": True,
    "datasets": [
        "cifake",
        "ai-generated-images-vs-real-images",
        ["cifake", "ai-generated-images-vs-real-images"],
    ],
    "split_strategy": {
        "images": "preserve_source_train_test_and_derive_val_from_train",
    },
    "target_threshold": 0.95,
}


PRIMARY_IMAGE_EXPERIMENTS = {
    "IMG-EXP-01": {
        "experiment_no": "IMG-EXP-01",
        "family": "ViT",
        "model_name": "vit_base_patch16_224",
        "params_note": "~86M",
        "reason": "Start the ViT ladder at the accepted legacy transformer floor.",
        "when_to_escalate": "Escalate to ViT-Large if ViT-Base is promising but below target.",
        "next_model_if_needed": "vit_large_patch16_224",
        "description": "Primary transformer baseline for image-only spatial training.",
    },
    "IMG-EXP-02": {
        "experiment_no": "IMG-EXP-02",
        "family": "ViT",
        "model_name": "vit_large_patch16_224",
        "params_note": "~307M",
        "reason": "Second-stage transformer escalation above the legacy floor.",
        "when_to_escalate": "Escalate to ViT-Huge only if ViT-Large is strong but still below target.",
        "next_model_if_needed": "vit_huge_patch14_224",
        "description": "High-capacity transformer follow-up above the legacy floor.",
    },
    "IMG-EXP-03": {
        "experiment_no": "IMG-EXP-03",
        "family": "ViT",
        "model_name": "vit_huge_patch14_224",
        "params_note": "~632M",
        "reason": "Maximum planned transformer-scale experiment if the family remains justified.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned transformer-scale image experiment.",
    },
    "IMG-EXP-04": {
        "experiment_no": "IMG-EXP-04",
        "family": "ConvNeXt",
        "model_name": "convnext_base",
        "params_note": "~89M",
        "reason": "Start the ConvNeXt ladder at the accepted legacy convolutional floor.",
        "when_to_escalate": "Escalate to ConvNeXt-Large if ConvNeXt-Base is competitive but below target.",
        "next_model_if_needed": "convnext_large",
        "description": "Primary modern convolutional baseline for image-only spatial learning.",
    },
    "IMG-EXP-05": {
        "experiment_no": "IMG-EXP-05",
        "family": "ConvNeXt",
        "model_name": "convnext_large",
        "params_note": "~198M",
        "reason": "Second-stage ConvNeXt escalation to test whether higher convolutional capacity helps.",
        "when_to_escalate": "Escalate to ConvNeXt-XL only if ConvNeXt-Large remains justified.",
        "next_model_if_needed": "convnext_xlarge",
        "description": "High-capacity ConvNeXt follow-up above the legacy floor.",
    },
    "IMG-EXP-06": {
        "experiment_no": "IMG-EXP-06",
        "family": "ConvNeXt",
        "model_name": "convnext_xlarge",
        "params_note": "~350M",
        "reason": "Maximum planned ConvNeXt-scale experiment if the family continues to justify scaling.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned ConvNeXt-scale image experiment.",
    },
    "IMG-EXP-07": {
        "experiment_no": "IMG-EXP-07",
        "family": "Swin",
        "model_name": "swin_base_patch4_window7_224",
        "params_note": "~88M",
        "reason": "Add a high-capacity hierarchical transformer above the 80M floor.",
        "when_to_escalate": "Escalate to Swin-Large if Swin-Base is promising but below target.",
        "next_model_if_needed": "swin_large_patch4_window7_224",
        "description": "Primary Swin transformer baseline for image-only spatial training.",
    },
    "IMG-EXP-08": {
        "experiment_no": "IMG-EXP-08",
        "family": "Swin",
        "model_name": "swin_large_patch4_window7_224",
        "params_note": "~197M",
        "reason": "Higher-capacity Swin follow-up above the 80M floor.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned Swin image experiment.",
    },
    "IMG-EXP-09": {
        "experiment_no": "IMG-EXP-09",
        "family": "DeiT",
        "model_name": "deit3_base_patch16_224",
        "params_note": "~86M",
        "reason": "Add a base-scale data-efficient transformer that still clears the 80M floor.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "DeiT-based transformer baseline for image-only spatial training.",
    },
    "IMG-EXP-10": {
        "experiment_no": "IMG-EXP-10",
        "family": "ConvNeXtV2",
        "model_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "params_note": "~89M",
        "reason": "Append a newer ConvNeXt family while staying above the 80M floor.",
        "when_to_escalate": "Escalate to ConvNeXtV2-Large if Base is promising but below target.",
        "next_model_if_needed": "convnextv2_large.fcmae_ft_in22k_in1k",
        "description": "Primary ConvNeXtV2 baseline for image-only spatial training.",
    },
    "IMG-EXP-11": {
        "experiment_no": "IMG-EXP-11",
        "family": "ConvNeXtV2",
        "model_name": "convnextv2_large.fcmae_ft_in22k_in1k",
        "params_note": "~198M",
        "reason": "Higher-capacity ConvNeXtV2 follow-up above the 80M floor.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned ConvNeXtV2 image experiment.",
    },
    "IMG-EXP-12": {
        "experiment_no": "IMG-EXP-12",
        "family": "MaxViT",
        "model_name": "maxvit_base_tf_224.in1k",
        "params_note": "~119M",
        "reason": "Add a hybrid high-capacity vision backbone that clears the 80M floor comfortably.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "MaxViT baseline for image-only spatial training.",
    },
    "IMG-EXP-13": {
        "experiment_no": "IMG-EXP-13",
        "family": "EVA",
        "model_name": "eva02_base_patch14_224.mim_in22k_ft_in22k_in1k",
        "params_note": "~86M",
        "reason": "Add an EVA base model that meets the 80M minimum while broadening the transformer pool.",
        "when_to_escalate": "Escalate to EVA-Large if EVA-Base is promising but below target.",
        "next_model_if_needed": "eva02_large_patch14_224.mim_m38m_ft_in22k_in1k",
        "description": "Primary EVA transformer baseline for image-only spatial training.",
    },
    "IMG-EXP-14": {
        "experiment_no": "IMG-EXP-14",
        "family": "EVA",
        "model_name": "eva02_large_patch14_224.mim_m38m_ft_in22k_in1k",
        "params_note": "~304M",
        "reason": "Higher-capacity EVA follow-up above the 80M floor.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned EVA image experiment.",
    },
}


FAMILY_ESCALATION_GUIDE = {
    "ViT": [
        {"model_name": "vit_base_patch16_224", "params_note": "~86M"},
        {"model_name": "vit_large_patch16_224", "params_note": "~307M"},
        {"model_name": "vit_huge_patch14_224", "params_note": "~632M"},
    ],
    "ConvNeXt": [
        {"model_name": "convnext_base", "params_note": "~89M"},
        {"model_name": "convnext_large", "params_note": "~198M"},
        {"model_name": "convnext_xlarge", "params_note": "~350M"},
    ],
    "Swin": [
        {"model_name": "swin_base_patch4_window7_224", "params_note": "~88M"},
        {"model_name": "swin_large_patch4_window7_224", "params_note": "~197M"},
    ],
    "DeiT": [
        {"model_name": "deit3_base_patch16_224", "params_note": "~86M"},
    ],
    "ConvNeXtV2": [
        {"model_name": "convnextv2_base.fcmae_ft_in22k_in1k", "params_note": "~89M"},
        {"model_name": "convnextv2_large.fcmae_ft_in22k_in1k", "params_note": "~198M"},
    ],
    "MaxViT": [
        {"model_name": "maxvit_base_tf_224.in1k", "params_note": "~119M"},
    ],
    "EVA": [
        {"model_name": "eva02_base_patch14_224.mim_in22k_ft_in22k_in1k", "params_note": "~86M"},
        {"model_name": "eva02_large_patch14_224.mim_m38m_ft_in22k_in1k", "params_note": "~304M"},
    ],
}


def get_image_experiment_registry():
    """Return the image experiment registry merged with global defaults."""
    registry = {}
    for exp_no, config in PRIMARY_IMAGE_EXPERIMENTS.items():
        merged = deepcopy(GLOBAL_IMAGE_DEFAULTS)
        merged.update(config)
        registry[exp_no] = merged
    return registry


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_image_experiment_registry())
