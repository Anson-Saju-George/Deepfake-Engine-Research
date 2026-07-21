"""Config-aware evaluation entry point for completed video runs.

This is a small compatibility layer around the established evaluator. It
replaces only the model loader so every saved temporal head is reconstructed
from ``config.json`` before evaluation.
"""

import torch

from train.video import test_video as _legacy
from train.video.video_train_backbone import DEVICE, TimmVideoClassifier


def load_video_model(config: dict, checkpoint_path):
    model = TimmVideoClassifier(config["model_name"], config=config).to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


_legacy.load_video_model = load_video_model

discover_video_runs = _legacy.discover_video_runs
evaluate_video_run = _legacy.evaluate_video_run
main = _legacy.main


if __name__ == "__main__":
    main()
