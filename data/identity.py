"""Explicit, per-dataset identity extraction for identity-aware splitting.

Design constraint (Current-Status.md S12 risk 8 / FIXTURE_PLAN.md T1.2):
the previous `DatasetBuilder.get_identity` silently treated any filename that
didn't match its two hardcoded heuristics as its own unique identity. That
quietly defeats identity-aware splitting for any file that doesn't match --
it doesn't error, it doesn't warn, the leak just happens.

This module replaces that with declared, per-dataset strategies. A strategy
either:
  - matches a known naming convention and returns one or more identity
    tokens for the file (a fake/face-swap video can legitimately involve
    two distinct identities -- source and target -- both of which must be
    kept out of opposite splits), or
  - is declared "no_shared_identity" for datasets/subsets where each file
    is genuinely its own independent source (verified on-disk, not assumed --
    see the per-dataset notes below), or
  - raises IdentityExtractionError for a file that matches none of the
    declared patterns for its dataset. This is a deliberate, loud failure --
    better to stop a split from being built than to silently under-group.

Ground truth used to write these patterns (verified against the actual
files on disk in this repo, not just the datasets' public documentation):

  faceforensics++/fake/Deepfakes/c23/videos/000_003.mp4        -> plain SRC_TGT
  faceforensics++/fake/DeepFakeDetection/c23/videos/01_02__exit_phone_room__YVGY8LOK.mp4
                                                                 -> actor-pair + scene + hash
  faceforensics++/real/youtube/c23/videos/000.mp4               -> plain ID
  faceforensics++/real/actors/c23/videos/01__exit_phone_room.mp4 -> actor + scene

  celeb-df-v2/Celeb-real/id0_0000.mp4                            -> single identity
  celeb-df-v2/Celeb-fake/id0_id16_0000.mp4                       -> TWO identities (source+target)
  celeb-df-v2/Celeb-fake/id16_id3_0005.noaudio.mp4               -> TWO identities, with a rare
                                                                     ".noaudio" pre-extension suffix
                                                                     (confirmed: 4 of 6533 celeb-df-v2
                                                                     files have this; Path.stem only
                                                                     strips ".mp4", so the plain regex
                                                                     doesn't match without stripping
                                                                     ".noaudio" first)
  celeb-df-v2/YouTube-real/00000.mp4                             -> no shared identity across files
                                                                     (independent YouTube source clips,
                                                                     confirmed: no "idN" token, no shared
                                                                     numbering scheme across files)

  real-ai-videos/{fake,real}/ai (N).mp4                          -> no shared identity across files
                                                                     (confirmed: filenames are a bare
                                                                     sequential counter with no subject
                                                                     encoding at all)
"""
from __future__ import annotations

import re
from pathlib import Path


class IdentityExtractionError(ValueError):
    """Raised when a filename matches none of its dataset's declared identity patterns."""


# ---------------------------------------------------------------- faceforensics++

_FFPP_PLAIN_FAKE = re.compile(r"^(\d{3})_(\d{3})$")           # 000_003
_FFPP_PLAIN_REAL = re.compile(r"^(\d{3})$")                   # 000
_FFPP_ACTOR_PAIR = re.compile(r"^(\d{1,3})_(\d{1,3})__.+$")   # 01_02__exit_phone_room__HASH
_FFPP_ACTOR_SOLO = re.compile(r"^(\d{1,3})__.+$")             # 01__exit_phone_room


def ffpp_identity(path: str) -> tuple[str, ...]:
    stem = Path(path).stem
    m = _FFPP_PLAIN_FAKE.match(stem)
    if m:
        return (f"ffpp:{m.group(1)}", f"ffpp:{m.group(2)}")
    m = _FFPP_ACTOR_PAIR.match(stem)
    if m:
        return (f"ffpp:{m.group(1)}", f"ffpp:{m.group(2)}")
    m = _FFPP_PLAIN_REAL.match(stem)
    if m:
        return (f"ffpp:{m.group(1)}",)
    m = _FFPP_ACTOR_SOLO.match(stem)
    if m:
        return (f"ffpp:{m.group(1)}",)
    raise IdentityExtractionError(
        f"faceforensics++ file does not match any declared naming convention: {path!r}"
    )


# FF++ manipulation-type detection, for LOMO (leave-one-manipulation-out, T2b).
# Verified against this repo's actual on-disk FF++ tree (Current-Status.md /
# this session's directory scan): only "Deepfakes" and "DeepFakeDetection" fake
# subsets currently exist here -- "Face2Face", "FaceSwap", "NeuralTextures" are
# NOT present in this corpus yet (they're part of the full FF++ release and the
# eventual corpus-rebuild plan in docs/RESEARCH_PLAN.md, just not downloaded here).
# A LOMO config for an absent manipulation would silently produce an empty test
# pool if not caught -- see lomo_manipulations_present() below, used to guard that.
FFPP_KNOWN_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]


def detect_ffpp_manipulation(path: str) -> str:
    """Return the manipulation type for a faceforensics++ file, or "real" for
    an unmanipulated video. Based on path components (dataset directory
    names), not filename -- these are stable across this repo's on-disk
    layout (fake/<Manipulation>/c23/videos/..., real/<youtube|actors>/c23/videos/...)."""
    parts = Path(path).parts
    for manipulation in FFPP_KNOWN_MANIPULATIONS:
        if manipulation in parts:
            return manipulation
    if "real" in parts:
        return "real"
    raise IdentityExtractionError(
        f"faceforensics++ file has no recognized manipulation-type directory component: {path!r}"
    )


# ---------------------------------------------------------------- celeb-df-v2

_CELEBDF_PAIR = re.compile(r"^(id\d+)_(id\d+)_\d+$")   # id0_id16_0000  (fake: source_target)
_CELEBDF_SOLO = re.compile(r"^(id\d+)_\d+$")           # id0_0000       (real)


def celebdf_identity(path: str) -> tuple[str, ...]:
    stem = Path(path).stem
    if stem.endswith(".noaudio"):
        stem = stem[: -len(".noaudio")]
    m = _CELEBDF_PAIR.match(stem)
    if m:
        return (f"celebdf:{m.group(1)}", f"celebdf:{m.group(2)}")
    m = _CELEBDF_SOLO.match(stem)
    if m:
        return (f"celebdf:{m.group(1)}",)
    raise IdentityExtractionError(
        f"celeb-df-v2 Celeb-real/Celeb-fake file does not match the idN[_idM]_NNNN "
        f"naming convention: {path!r}"
    )


# ---------------------------------------------------------- no-shared-identity datasets

def per_file_identity(path: str) -> tuple[str, ...]:
    """For datasets/subsets where each file is verified to be its own independent
    source (celeb-df-v2/YouTube-real, real-ai-videos). Never raises -- there is
    no pattern to fail to match, the file *is* the identity, by design."""
    return (f"file:{Path(path).resolve()}",)


# ---------------------------------------------------------------- dataset -> strategy

# Maps (dataset_name, subset_hint) to a strategy function. subset_hint is matched
# against the file's path components (case-insensitive) so a single dataset can
# mix conventions, as faceforensics++ does. First matching hint wins; datasets
# with only one convention use subset_hint=None to match unconditionally.
IDENTITY_STRATEGIES: dict[str, list[tuple[str | None, callable]]] = {
    "faceforensics++": [
        (None, ffpp_identity),  # ffpp_identity itself handles both sub-conventions
    ],
    "celeb-df-v2": [
        ("youtube-real", per_file_identity),  # verified: no shared identity across files
        (None, celebdf_identity),             # Celeb-real / Celeb-fake
    ],
    "real-ai-videos": [
        (None, per_file_identity),  # verified: bare sequential counter, no subject encoding
    ],
}


def extract_identities(path: str, dataset_name: str) -> tuple[str, ...]:
    """Return one or more identity tokens for `path` under `dataset_name`'s
    declared strategy. Raises IdentityExtractionError if the dataset has a
    declared strategy but the file matches none of it, and raises KeyError
    if the dataset has no declared strategy at all (fail loud on datasets
    nobody has vetted yet, rather than silently falling back to per-file)."""
    if dataset_name not in IDENTITY_STRATEGIES:
        raise KeyError(
            f"No identity strategy declared for dataset {dataset_name!r}. "
            f"Add one to data/identity.py:IDENTITY_STRATEGIES before using this "
            f"dataset in identity-aware splitting -- do not fall back silently."
        )

    path_lower = str(path).lower().replace("\\", "/")
    for subset_hint, strategy_fn in IDENTITY_STRATEGIES[dataset_name]:
        if subset_hint is None or subset_hint in path_lower:
            return strategy_fn(path)

    raise IdentityExtractionError(
        f"No identity strategy matched for dataset {dataset_name!r}, path {path!r}"
    )
