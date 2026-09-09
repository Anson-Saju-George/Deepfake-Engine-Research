"""Declarative dataset metadata layered on top of data/dataloader.py:DATASET_CONFIG.

FIXTURE_PLAN.md T1.1 asked for "adding a new dataset must be a declarative
entry, not new branching logic." DATASET_CONFIG in data/dataloader.py already
covers discovery (root layout, class->label mapping) and is left untouched
here to avoid any risk to the existing scan_images()/scan_videos() logic.
This module adds the metadata DATASET_CONFIG doesn't carry: which identity
strategy applies (data/identity.py) and what role the dataset plays in a
train/test protocol.

Role enforcement is DELIBERATELY NOT WIRED IN as a hard constraint yet. Per
docs/RESEARCH_PLAN.md's "Dataset roles (never violate)" table, Celeb-DF v2 is
meant to be "TEST only, permanently" in the rebuilt corpus this refactor is
building toward. But the CURRENT active dataloader -- and all 22 completed
runs -- use celeb-df-v2 as part of an identity-aware 70/10/20 train/val/test
split, not test-only. Hard-enforcing the research-plan role now would silently
invalidate how every existing completed run used this dataset. That is a real
methodology decision (switch celeb-df-v2 to test-only), not a refactor detail,
and it belongs with the corpus rebuild (docs/RESEARCH_PLAN.md Phase 1), not
smuggled into this pass. `role` below is therefore descriptive/documented,
not enforced -- `enforce_roles=True` in data/splits.py-consuming code is left
for that future, explicit decision.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRegistryEntry:
    name: str
    media_type: str            # "image" | "video"
    identity_strategy: str     # key into data/identity.py:IDENTITY_STRATEGIES, or "none" (images)
    role: str                  # "train_only" | "test_only" | "either" -- see module docstring: descriptive, not enforced yet
    role_source: str           # where this role claim comes from, so it's auditable


DATASET_REGISTRY: dict[str, DatasetRegistryEntry] = {
    "cifake": DatasetRegistryEntry(
        name="cifake", media_type="image", identity_strategy="none", role="either",
        role_source="preserves source train/test split; no video-identity concept applies",
    ),
    "ai-generated-images-vs-real-images": DatasetRegistryEntry(
        name="ai-generated-images-vs-real-images", media_type="image", identity_strategy="none", role="either",
        role_source="preserves source train/test split; no video-identity concept applies",
    ),
    "celeb-df-v2": DatasetRegistryEntry(
        name="celeb-df-v2", media_type="video", identity_strategy="celeb-df-v2", role="either",
        role_source=(
            "CURRENT active behavior: identity-aware 70/10/20 split (all 22 completed video runs). "
            "docs/RESEARCH_PLAN.md's dataset-roles table instead states 'TEST only, permanently' for "
            "the REBUILT corpus -- that is a stated future decision, not yet enforced. See module docstring."
        ),
    ),
    "faceforensics++": DatasetRegistryEntry(
        name="faceforensics++", media_type="video", identity_strategy="faceforensics++", role="either",
        role_source=(
            "CURRENT active behavior: identity-aware 70/10/20 split. docs/RESEARCH_PLAN.md's dataset-roles "
            "table states 'TRAIN only' for the REBUILT corpus -- not yet enforced. See module docstring."
        ),
    ),
    "real-ai-videos": DatasetRegistryEntry(
        name="real-ai-videos", media_type="video", identity_strategy="real-ai-videos", role="either",
        role_source="small auxiliary dataset; no stated role constraint in docs/RESEARCH_PLAN.md",
    ),
}


def get_entry(dataset_name: str) -> DatasetRegistryEntry:
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset {dataset_name!r} has no registry entry. Add one to "
            f"data/dataset_registry.py:DATASET_REGISTRY before using it in "
            f"identity-aware splitting."
        )
    return DATASET_REGISTRY[dataset_name]
