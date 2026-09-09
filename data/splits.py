"""Identity-disjoint corpus construction.

Promoted from docs/code/splits.py (FIXTURE_PLAN.md T1.3), and extended in one
material way: docs/code/splits.py's VideoRecord carried a single `identity`
string. Real face-swap datasets don't work that way -- a Celeb-DF fake video
`id0_id16_0000.mp4` involves BOTH id0 (source face) and id16 (target video),
and a FaceForensics++ DeepFakeDetection file `01_02__scene__hash.mp4` is the
same shape. If only one of the two identities is tracked (as the original
module did), the untracked identity can still leak across splits through a
different record. This module tracks a record's full identity set and groups
records by connected components over the identity-conflict graph (union-find):
if id0 and id16 must stay together (same record), and id16 and id23 must stay
together (a different record), then id0/id16/id23 all end up in one component
and are assigned to the same split as a unit. This is the only correct way to
guarantee no shared identity crosses a split boundary in a face-swap corpus.

Three jobs, same as the original:
  1. Identity-disjoint splits (no subject in two partitions) -- now correct
     for multi-identity records, not just single-identity ones.
  2. Balance WITHOUT discarding data, via asymmetric clip sampling.
  3. Leave-one-manipulation-out configs -> the "unseen generator" axis.

Identity extraction itself lives in data/identity.py (per-dataset declared
strategies, fail loud on unmatched filenames -- see that module's docstring
for the on-disk naming conventions this was verified against).
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict

from data.identity import extract_identities, IdentityExtractionError

FFPP_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def identities_of(path: str, dataset: str) -> tuple[str, ...]:
    """Thin wrapper over data/identity.py so callers here don't need to
    import both modules. Deliberately does not catch IdentityExtractionError
    or KeyError -- an unmatched/undeclared identity strategy should stop the
    split from being built, not be silently absorbed."""
    return extract_identities(path, dataset)


# ---------------------------------------------------------------- records
@dataclass
class VideoRecord:
    path: str
    label: int                     # 1 = real, 0 = fake -- matches data/dataloader.py:DATASET_CONFIG's
                                    # active convention exactly (build_record() is fed r["label"] straight
                                    # from DatasetBuilder, so this MUST match that, not docs/code/metrics.py's
                                    # opposite fake=1/real=0 convention -- an earlier version of this file
                                    # inherited the wrong one from docs/code/splits.py during promotion and
                                    # every real/fake-branching function below was silently backwards until
                                    # that was caught and fixed. See FIXTURE_PLAN.md S2 issue 2 for the
                                    # broader repo-wide convention conflict this file must NOT reintroduce.
    identities: tuple[str, ...]    # one or more identity tokens; must never cross a split boundary as a set
    dataset: str
    manipulation: str = "real"
    split: str = "train"
    n_clips: int = 1                # set by asymmetric sampling

    def as_dict(self):
        d = asdict(self)
        d["identities"] = list(self.identities)
        return d


def build_record(path: str, label: int, dataset: str, manipulation: str = "real") -> VideoRecord:
    return VideoRecord(
        path=path,
        label=label,
        identities=identities_of(path, dataset),
        dataset=dataset,
        manipulation=manipulation,
    )


# ------------------------------------------------------- Celeb-DF-v2 official test list
# datasets/videos/celeb-df-v2/List_of_testing_videos.txt is Celeb-DF-v2's official
# published test split (517 entries: 178 real, 340 fake -- verified against the
# actual file). It was never referenced by any code in this repo before this
# module. This repo's local copy of the dataset renames the official
# "Celeb-synthesis" directory to "Celeb-fake" -- confirmed by checking that the
# list's filenames resolve on disk under Celeb-fake/, not Celeb-synthesis/.
CELEBDF_TEST_LIST_DIR_REMAP = {"Celeb-synthesis": "Celeb-fake"}


def load_celebdf_official_test_list(path) -> set[tuple[str, str]]:
    """Parse List_of_testing_videos.txt. Format per line: '<label> <dir>/<file>',
    label 1=real / 0=fake (verified: all 340 label=0 lines are under
    Celeb-synthesis, all 178 label=1 lines are under Celeb-real/YouTube-real --
    matches this dataset's published 178R/340F test-set composition exactly).
    Returns {(dir_name, filename)} with dir_name already remapped to this
    repo's actual on-disk directory name, so callers can match by
    (Path(video_path).parent.name, Path(video_path).name) without caring
    about OS path separators or the official/local directory-naming mismatch.
    """
    entries = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        _label, rel = line.split(None, 1)
        parts = rel.replace("\\", "/").split("/")
        dir_name, filename = parts[-2], parts[-1]
        dir_name = CELEBDF_TEST_LIST_DIR_REMAP.get(dir_name, dir_name)
        entries.add((dir_name, filename))
    return entries


def is_in_celebdf_official_test(path, official_entries: set[tuple[str, str]]) -> bool:
    p = Path(path)
    return (p.parent.name, p.name) in official_entries


# ---------------------------------------------------------------- FF++ split
def load_ffpp_official_split(json_dir):
    """Load FF++ train/val/test .json (each a list of ["000","003"] pairs)."""
    out = {}
    for sp in ("train", "val", "test"):
        p = Path(json_dir) / f"{sp}.json"
        if not p.exists():
            return None
        ids = set()
        for pair in json.loads(p.read_text()):
            ids.update(str(x) for x in pair)
        out[sp] = ids
    return out


# ---------------------------------------------------------------- union-find grouping
class _UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def group_by_connected_identities(records: list[VideoRecord]) -> dict[str, list[VideoRecord]]:
    """Group records so that any two identities ever co-occurring in the same
    record end up in the same group, transitively. Returns {component_key: [records]}."""
    uf = _UnionFind()
    for record in records:
        ids = record.identities
        for identity in ids:
            uf.find(identity)  # register
        for a, b in zip(ids, ids[1:]):
            uf.union(a, b)

    groups: dict[str, list[VideoRecord]] = {}
    for record in records:
        # any identity in the record resolves to the same root after union()
        root = uf.find(record.identities[0])
        groups.setdefault(root, []).append(record)
    return groups


def deterministic_split(group_keys, ratios=(0.7, 0.1, 0.2), seed=42):
    """Stable hash-based split over connected-component group keys.
    Reproducible without extra files."""
    keys = sorted(set(group_keys))
    keyed = sorted(keys, key=lambda i: hashlib.md5(f"{seed}:{i}".encode()).hexdigest())
    n = len(keyed)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    return {"train": set(keyed[:n_tr]),
            "val": set(keyed[n_tr:n_tr + n_va]),
            "test": set(keyed[n_tr + n_va:])}


def assign_splits(records: list[VideoRecord], seed: int = 42, ratios=(0.7, 0.1, 0.2)) -> list[VideoRecord]:
    """Assign every record a split, guaranteeing no identity crosses a split
    boundary even when it's shared across otherwise-unrelated records."""
    groups = group_by_connected_identities(records)
    split_map = deterministic_split(groups.keys(), ratios=ratios, seed=seed)
    inv = {key: sp for sp, keys in split_map.items() for key in keys}

    assigned = []
    for group_key, group_records in groups.items():
        split = inv[group_key]
        for record in group_records:
            record.split = split
            assigned.append(record)
    return assigned


# ------------------------------------------------- asymmetric clip sampling
def balance_by_clip_sampling(records, split="train", max_clips=8, verbose=True):
    """Balance real vs fake at the SAMPLE level without dropping any video.

    Real videos get more clips each; fake videos get fewer (or vice versa).
    This is how the literature handles a dataset's video-level class skew
    while keeping all data.
    """
    sub = [r for r in records if r.split == split]
    n_real = sum(1 for r in sub if r.label == 1)
    n_fake = sum(1 for r in sub if r.label == 0)
    if n_real == 0 or n_fake == 0:
        return records

    if n_fake >= n_real:
        clips_real = min(max(round(n_fake / n_real), 1), max_clips)
        clips_fake = 1
    else:
        clips_real = 1
        clips_fake = min(max(round(n_real / n_fake), 1), max_clips)

    for r in sub:
        r.n_clips = clips_real if r.label == 1 else clips_fake

    if verbose:
        er, ef = n_real * clips_real, n_fake * clips_fake
        print(f"[balance:{split}] videos {n_real}R/{n_fake}F -> "
              f"clips {er}R/{ef}F ({100*er/(er+ef):.1f}% real) "
              f"[{clips_real} clips/real, {clips_fake} clips/fake]")
    return records


def expand_by_n_clips(records: list[VideoRecord]) -> list[tuple[str, int, str]]:
    """Turn each record's n_clips into that many (path, label, dtype) sample
    tuples, the shape DeepFakeDataset already consumes. Each repetition of a
    video's path is a distinct dataset-list entry; because DeepFakeDataset's
    _contiguous_indices() picks a fresh random start offset on every
    __getitem__ call (not a fixed index computed once), repeating a path
    n_clips times already yields n_clips different sampled clips per epoch
    without any change to DeepFakeDataset itself."""
    samples = []
    for record in records:
        for _ in range(max(1, record.n_clips)):
            samples.append((record.path, record.label, "video"))
    return samples


# ------------------------------------------------- leave-one-manipulation-out
def lomo_configs(manipulations=None):
    """Generate the unseen-generator protocol: train on N-1, test on held-out."""
    ms = manipulations or FFPP_MANIPULATIONS
    return [{"name": f"LOMO_{h}", "train_manipulations": [m for m in ms if m != h],
             "test_manipulation": h} for h in ms]


def manipulations_present(records) -> set[str]:
    """Which manipulation types actually have >=1 record in this set. Use
    this to filter lomo_configs() before running one -- a config whose
    test_manipulation isn't in this set would silently produce an empty
    unseen-test pool rather than erroring, which is exactly the kind of
    quiet failure this refactor exists to avoid."""
    return {r.manipulation for r in records if r.label == 0}


def usable_lomo_configs(records, manipulations=None):
    """LOMO configs rebuilt against the manipulations ACTUALLY present in
    `records`, not the nominal FFPP_MANIPULATIONS list. Two distinct fixes
    over calling lomo_configs() directly:

      1. Configs whose test_manipulation isn't present at all are dropped
         (e.g. Face2Face/FaceSwap/NeuralTextures, not yet downloaded into
         this repo -- see data/identity.py:FFPP_KNOWN_MANIPULATIONS) --
         otherwise they'd silently produce an empty unseen-test pool.

      2. train_manipulations is rebuilt as "present manipulations minus the
         held-out one", not "the nominal 4 minus the held-out one". This
         matters concretely: this corpus's fake videos are split across
         Deepfakes AND DeepFakeDetection (a distinct FF++ release variant,
         not one of the 4 standard manipulations). Using the nominal list's
         train_manipulations for LOMO_Deepfakes would silently exclude every
         DeepFakeDetection fake video from the training pool too -- not
         held out as unseen, not trained on, just dropped. Caught by
         checking the actual train-pool size against expectations before
         this fix landed.
    """
    present = manipulations_present(records)
    configs = lomo_configs(manipulations)
    usable = []
    skipped = []
    for cfg in configs:
        if cfg["test_manipulation"] not in present:
            skipped.append(cfg["test_manipulation"])
            continue
        rebuilt = dict(cfg)
        rebuilt["train_manipulations"] = sorted(present - {cfg["test_manipulation"]})
        usable.append(rebuilt)
    if skipped:
        print(f"[lomo] skipping configs for manipulations not present in this corpus: {skipped}")
    return usable


def filter_for_lomo(records, cfg):
    """Split records into (train_pool, unseen_test_pool) for one LOMO config."""
    tr, te = [], []
    for r in records:
        if r.label == 1:  # real: no manipulation type to filter on, bypass unconditionally
            (tr if r.split in ("train", "val") else te).append(r)
        elif r.manipulation in cfg["train_manipulations"]:
            if r.split in ("train", "val"):
                tr.append(r)
        elif r.manipulation == cfg["test_manipulation"]:
            te.append(r)
    return tr, te


# ---------------------------------------------------------------- reporting
def summarize(records):
    rows = {}
    for r in records:
        k = (r.dataset, r.split)
        d = rows.setdefault(k, {"real": 0, "fake": 0, "clips_real": 0, "clips_fake": 0,
                                "identities": set()})
        d["identities"].update(r.identities)
        if r.label == 1:
            d["real"] += 1; d["clips_real"] += r.n_clips
        else:
            d["fake"] += 1; d["clips_fake"] += r.n_clips
    lines = [f"{'dataset':<12}{'split':<7}{'real':>7}{'fake':>7}{'%real':>7}"
             f"{'clipR':>8}{'clipF':>8}{'%clipR':>8}{'ids':>6}"]
    for (ds, sp), d in sorted(rows.items()):
        tot = d["real"] + d["fake"]; ct = d["clips_real"] + d["clips_fake"]
        lines.append(f"{ds:<12}{sp:<7}{d['real']:>7}{d['fake']:>7}"
                     f"{100*d['real']/max(tot,1):>6.1f}%"
                     f"{d['clips_real']:>8}{d['clips_fake']:>8}"
                     f"{100*d['clips_real']/max(ct,1):>7.1f}%{len(d['identities']):>6}")
    return "\n".join(lines)


def check_leakage(records: list[VideoRecord]) -> list[tuple[str, str, str, str]]:
    """Hard assertion: no identity may appear in more than one split.
    Checks every identity in every record's identity set, not just one."""
    seen: dict[str, str] = {}
    bad = []
    for r in records:
        for identity in r.identities:
            key = f"{r.dataset}:{identity}"
            if key in seen and seen[key] != r.split:
                bad.append((r.dataset, identity, seen[key], r.split))
            seen[key] = r.split
    return bad


def save_manifest(records, path):
    Path(path).write_text(json.dumps([r.as_dict() for r in records], indent=1))
    return path
