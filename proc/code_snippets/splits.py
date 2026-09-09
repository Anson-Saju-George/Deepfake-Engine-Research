"""
dfx.splits — corpus construction that fixes the imbalance + leakage problems.

Three jobs:
  1. Identity-disjoint splits (no subject in two partitions).
  2. Balance WITHOUT discarding data, via asymmetric clip sampling
     (the field's actual solution: sample ~4x more clips from each real
      video, since FF++ ships 1 real : 4 fake once manipulations are pooled).
  3. Leave-one-manipulation-out configs -> the "unseen generator" axis.

Naming conventions handled:
  FF++    real  "000.mp4"           -> identity "000"
          fake  "000_003.mp4"       -> identity "000"  (target sequence)
  Celeb-DF real "id0_0000.mp4"      -> identity "id0"
          fake  "id0_id1_0000.mp4"  -> identity "id0"
  fallback: full stem as identity (never silently merge unknown files)
"""
from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict

FFPP_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ---------------------------------------------------------------- identity
def ffpp_identity(name: str) -> str:
    s = Path(name).stem
    m = re.fullmatch(r"(\d{3})(?:_(\d{3}))?", s)
    return m.group(1) if m else s


def celebdf_identity(name: str) -> str:
    s = Path(name).stem
    m = re.match(r"(id\d+)", s)
    return m.group(1) if m else s


IDENTITY_FN = {"ffpp": ffpp_identity, "celebdf": celebdf_identity}


def identity_of(name, dataset="ffpp"):
    return IDENTITY_FN.get(dataset, lambda n: Path(n).stem)(name)


# ---------------------------------------------------------------- records
@dataclass
class VideoRecord:
    path: str
    label: int                 # 1 = fake, 0 = real
    identity: str
    dataset: str
    manipulation: str = "real"
    split: str = "train"
    n_clips: int = 1           # set by asymmetric sampling

    def as_dict(self):
        return asdict(self)


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


def deterministic_split(identities, ratios=(0.7, 0.1, 0.2), seed=42):
    """Stable hash-based identity split; reproducible without extra files."""
    ids = sorted(set(identities))
    keyed = sorted(ids, key=lambda i: hashlib.md5(f"{seed}:{i}".encode()).hexdigest())
    n = len(keyed)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    return {"train": set(keyed[:n_tr]),
            "val": set(keyed[n_tr:n_tr + n_va]),
            "test": set(keyed[n_tr + n_va:])}


def assign_splits(records, split_map):
    inv = {i: sp for sp, ids in split_map.items() for i in ids}
    kept = []
    for r in records:
        sp = inv.get(r.identity)
        if sp is not None:
            r.split = sp
            kept.append(r)
    return kept


# ------------------------------------------------- asymmetric clip sampling
def balance_by_clip_sampling(records, split="train", max_clips=8, verbose=True):
    """Balance real vs fake at the SAMPLE level without dropping any video.

    Real videos get more clips each; fake videos get fewer. This is how the
    literature handles FF++'s 1:4 video-level skew while keeping all data.
    """
    sub = [r for r in records if r.split == split]
    n_real = sum(1 for r in sub if r.label == 0)
    n_fake = sum(1 for r in sub if r.label == 1)
    if n_real == 0 or n_fake == 0:
        return records

    if n_fake >= n_real:
        clips_real = min(max(round(n_fake / n_real), 1), max_clips)
        clips_fake = 1
    else:
        clips_real = 1
        clips_fake = min(max(round(n_real / n_fake), 1), max_clips)

    for r in sub:
        r.n_clips = clips_real if r.label == 0 else clips_fake

    if verbose:
        er, ef = n_real * clips_real, n_fake * clips_fake
        print(f"[balance:{split}] videos {n_real}R/{n_fake}F -> "
              f"clips {er}R/{ef}F ({100*er/(er+ef):.1f}% real) "
              f"[{clips_real} clips/real, {clips_fake} clips/fake]")
    return records


# ------------------------------------------------- leave-one-manipulation-out
def lomo_configs(manipulations=None):
    """Generate the unseen-generator protocol: train on N-1, test on held-out."""
    ms = manipulations or FFPP_MANIPULATIONS
    return [{"name": f"LOMO_{h}", "train_manipulations": [m for m in ms if m != h],
             "test_manipulation": h} for h in ms]


def filter_for_lomo(records, cfg):
    """Split records into (train_pool, unseen_test_pool) for one LOMO config."""
    tr, te = [], []
    for r in records:
        if r.label == 0:
            tr.append(r) if r.split in ("train", "val") else te.append(r)
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
        d["identities"].add(r.identity)
        if r.label == 0:
            d["real"] += 1; d["clips_real"] += r.n_clips
        else:
            d["fake"] += 1; d["clips_fake"] += r.n_clips
    lines = [f"{'dataset':<10}{'split':<7}{'real':>7}{'fake':>7}{'%real':>7}"
             f"{'clipR':>8}{'clipF':>8}{'%clipR':>8}{'ids':>6}"]
    for (ds, sp), d in sorted(rows.items()):
        tot = d["real"] + d["fake"]; ct = d["clips_real"] + d["clips_fake"]
        lines.append(f"{ds:<10}{sp:<7}{d['real']:>7}{d['fake']:>7}"
                     f"{100*d['real']/max(tot,1):>6.1f}%"
                     f"{d['clips_real']:>8}{d['clips_fake']:>8}"
                     f"{100*d['clips_real']/max(ct,1):>7.1f}%{len(d['identities']):>6}")
    return "\n".join(lines)


def check_leakage(records):
    """Hard assertion: no identity may appear in more than one split."""
    seen = {}
    bad = []
    for r in records:
        key = (r.dataset, r.identity)
        if key in seen and seen[key] != r.split:
            bad.append((r.dataset, r.identity, seen[key], r.split))
        seen[key] = r.split
    return bad


def save_manifest(records, path):
    Path(path).write_text(json.dumps([r.as_dict() for r in records], indent=1))
    return path
