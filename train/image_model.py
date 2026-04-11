"""Image experiment registry for the new top-level image pipeline.

This file keeps the primary image-only experiment ladder explicit. The chosen
families and sizes are intentionally above the legacy lower bound already used
in the repo.

Implemented status:
- smoke test exists in ``train.image_simulate``
- ViT trainer/runner currently exist for ``IMG-EXP-01..03``
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
        "family": "EfficientNet",
        "model_name": "tf_efficientnet_b6",
        "params_note": "~43M",
        "reason": "Start the EfficientNet ladder above the older B4 floor.",
        "when_to_escalate": "Escalate to EfficientNet-B7 if B6 is strong but still below target.",
        "next_model_if_needed": "tf_efficientnet_b7",
        "description": "High-capacity efficient CNN baseline for image-only experiments.",
    },
    "IMG-EXP-08": {
        "experiment_no": "IMG-EXP-08",
        "family": "EfficientNet",
        "model_name": "tf_efficientnet_b7",
        "params_note": "~66M",
        "reason": "Maximum planned EfficientNet-scale experiment for the family.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned EfficientNet-scale image experiment.",
    },
    "IMG-EXP-09": {
        "experiment_no": "IMG-EXP-09",
        "family": "ResNet",
        "model_name": "resnet101.a1h_in1k",
        "params_note": "~44.5M",
        "reason": "Start the ResNet ladder above the old ResNet-50 floor.",
        "when_to_escalate": "Escalate to ResNet-152 if ResNet-101 is stable but below target.",
        "next_model_if_needed": "resnet152.a1_in1k",
        "description": "High-capacity classical CNN baseline for image-only experiments.",
    },
    "IMG-EXP-10": {
        "experiment_no": "IMG-EXP-10",
        "family": "ResNet",
        "model_name": "resnet152.a1_in1k",
        "params_note": "~60.2M",
        "reason": "Maximum planned ResNet-scale experiment if the family still warrants scaling.",
        "when_to_escalate": "No automatic escalation beyond this stage.",
        "next_model_if_needed": None,
        "description": "Maximum planned ResNet-scale image experiment.",
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
    "EfficientNet": [
        {"model_name": "tf_efficientnet_b6", "params_note": "~43M"},
        {"model_name": "tf_efficientnet_b7", "params_note": "~66M"},
    ],
    "ResNet": [
        {"model_name": "resnet101.a1h_in1k", "params_note": "~44.5M"},
        {"model_name": "resnet152.a1_in1k", "params_note": "~60.2M"},
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
