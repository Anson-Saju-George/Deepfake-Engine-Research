#!/usr/bin/env python3
r"""Upload completed run artifacts straight from train/ to Hugging Face Hub.

No staging directory. Earlier versions of this project built a separate
models/ copy first (via model_packer.py) and uploaded that; both are gone
now -- maintaining a full local duplicate of what's already in train/ was
redundant, and model_packer.py existed only to build it. This walks
train/image/ and train/video/ directly, and for each completed run (a
directory containing final_summary.json) uploads exactly three files
straight from their original location via HfApi.upload_file():

    best.pth        -- NOT last.pth, NOT the duplicate final_*.pth copy
    config.json
    final_summary.json

No .py, no .md (per request -- this also means run_record.md is excluded,
same as any other .md file; it's regenerable from config.json/final_summary.json
if ever needed), no training logs, no per-sample prediction CSVs (research-audit
artifacts, not needed by a serving backend).

Usage:
    python huggingface_uploader.py            # upload
    python huggingface_uploader.py --list     # show what would be uploaded, no network calls
    python huggingface_uploader.py --verify   # re-download a sample and hash-check after upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

REPO_ID = "Anson-Saju-George/deepfake-model-weights"
REPO_TYPE = "model"
PRIVATE_REPO = False
REPO_ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = REPO_ROOT / "train"

# Recommended: keep this as None and set the token in the environment:
#   $env:HF_TOKEN = "hf_your_token_here"        (PowerShell)
#   export HF_TOKEN=hf_your_token_here            (WSL/bash)
HF_TOKEN = None

COMMIT_MESSAGE = "Upload model weights directly from train/ (best.pth + config + final_summary only)"

INCLUDE_FILENAMES = ("config.json", "final_summary.json")  # best.pth resolved separately, see below


def sanitize_component(value: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or fallback


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunFiles:
    def __init__(self, summary_path: Path):
        self.source_dir = summary_path.parent
        self.summary = load_json(summary_path)
        config_path = self.source_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json next to {summary_path}")

        # Always resolve directly (self.source_dir / "best.pth"), never from the
        # final_summary.json "best_checkpoint" field. That field was written on
        # Windows and contains backslash-separated paths (e.g.
        # "train\image\ConvNeXt\...\best.pth"); under WSL/Linux, pathlib's POSIX
        # flavor does NOT treat "\" as a separator, so Path(checkpoint_value)
        # silently produces a path that never resolves. Caught in production:
        # 16 of 22 runs were skipped with "Missing checkpoint" under WSL before
        # this fix -- the only runs that worked were the ones where
        # best_checkpoint happened to be absent from final_summary.json,
        # falling through to this same direct-resolution path by accident.
        checkpoint_path = self.source_dir / "best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {summary_path}")

        parts = summary_path.parts
        train_index = parts.index("train")
        self.domain = parts[train_index + 1]  # "image" | "video"
        config = load_json(config_path)
        self.family = str(config.get("family") or self.source_dir.parent.name)
        self.model_name = str(config.get("model_name") or self.summary.get("model_name") or self.source_dir.name)
        self.experiment_no = str(self.summary.get("experiment_no") or config.get("experiment_no") or self.source_dir.name)
        self.run_name = sanitize_component(str(self.summary.get("run_name") or config.get("run_name") or self.source_dir.name))
        self.metrics = self.summary.get("test_metrics", {}) or {}

        self.local_files: dict[str, Path] = {
            "best.pth": checkpoint_path,
            "config.json": config_path,
            "final_summary.json": summary_path,
        }

    @property
    def remote_dir(self) -> str:
        family_model = sanitize_component(f"{self.family}_{self.model_name}")
        return f"{self.domain.lower()}/{family_model}/{self.run_name}"

    def remote_paths(self) -> dict[str, Path]:
        return {f"{self.remote_dir}/{name}": local_path for name, local_path in self.local_files.items()}

    def describe(self) -> str:
        acc = self.metrics.get("acc", self.summary.get("best_val_acc"))
        f1 = self.metrics.get("f1", self.summary.get("best_val_f1"))
        return f"{self.experiment_no} | {self.domain} | {self.family} | {self.model_name} | acc={acc} | f1={f1} -> {self.remote_dir}"


def discover_completed_runs() -> list[RunFiles]:
    runs = []
    for summary_path in sorted(TRAIN_ROOT.rglob("final_summary.json")):
        try:
            runs.append(RunFiles(summary_path))
        except Exception as exc:
            print(f"[skip] {summary_path}: {exc}")
    return runs


def build_readme(repo_id: str, run_count: int) -> str:
    return f"""---
library_name: pytorch
tags:
- deepfake-detection
- image-classification
- video-classification
---

# Deepfake Detection Model Weights

Model checkpoints for {run_count} completed image and video experiments, uploaded directly from
the source repo's `train/` tree (see `huggingface_uploader.py` in the source repo). No local
staging copy is maintained -- this repo IS the distribution artifact.

## Layout

```
image/<Family>_<model_name>/<run_name>/best.pth
image/<Family>_<model_name>/<run_name>/config.json
image/<Family>_<model_name>/<run_name>/final_summary.json
video/<Family>_<model_name>/<run_name>/...  (same shape)
```

Only the best checkpoint per run (not `last.pth`, not the duplicate `final_*.pth` copy that
exists in the source repo). No training logs, no per-sample prediction CSVs, no `.md` files.

## Pulling from a backend

```python
from huggingface_hub import snapshot_download

# everything
local_dir = snapshot_download(repo_id="{repo_id}")

# just one model (avoids pulling the whole multi-GB tree)
local_dir = snapshot_download(
    repo_id="{repo_id}",
    allow_patterns=["image/ConvNeXt_convnext_base/*"],
)
```
"""


def list_runs() -> list[RunFiles]:
    runs = discover_completed_runs()
    for run in runs:
        print(run.describe())
    total_files = sum(len(r.local_files) for r in runs)
    total_bytes = sum(p.stat().st_size for r in runs for p in r.local_files.values())
    print(f"\n{len(runs)} runs, {total_files} files, {total_bytes / (1024**3):.2f} GB")
    return runs


def upload(token: str | None = None) -> None:
    token = token or HF_TOKEN or os.environ.get("HF_TOKEN")
    if not REPO_ID or "/" not in REPO_ID:
        raise SystemExit("Set REPO_ID to something like: username/repo-name")
    if not token:
        raise SystemExit('Missing token. Set HF_TOKEN in the environment (see this file\'s docstring).')

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install huggingface_hub") from exc

    runs = discover_completed_runs()
    if not runs:
        raise SystemExit(f"No completed runs found under {TRAIN_ROOT}")

    api = HfApi(token=token)
    print(f"Creating or reusing {REPO_TYPE} repo: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=PRIVATE_REPO, exist_ok=True)

    total_files = sum(len(r.remote_paths()) for r in runs)
    uploaded = 0
    for run in runs:
        print(f"\n{run.describe()}")
        for remote_path, local_path in run.remote_paths().items():
            print(f"  uploading {remote_path} ({local_path.stat().st_size / (1024**2):.1f} MB)")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=COMMIT_MESSAGE,
            )
            uploaded += 1
            print(f"  [{uploaded}/{total_files}] done")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix="_README.md", delete=False) as handle:
        handle.write(build_readme(REPO_ID, len(runs)))
        readme_path = Path(handle.name)
    try:
        print("\nUploading README.md")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Update model card",
        )
    finally:
        readme_path.unlink(missing_ok=True)

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


def verify(token: str | None = None, sample_size: int = 5) -> None:
    """Download a sample of uploaded files and hash-check against the
    original train/ files -- proof the upload round-tripped correctly, not
    just that the API calls returned success."""
    token = token or HF_TOKEN or os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install huggingface_hub") from exc

    runs = discover_completed_runs()
    all_pairs = [(remote, local) for run in runs for remote, local in run.remote_paths().items()]
    if not all_pairs:
        raise SystemExit("No local run files found to verify against.")

    sample = all_pairs[:: max(1, len(all_pairs) // sample_size)][:sample_size]
    print(f"Verifying {len(sample)} of {len(all_pairs)} files (round-trip download + hash check)...")

    mismatches = []
    for remote_path, local_path in sample:
        print(f"  downloading {remote_path} ...")
        downloaded_path = Path(
            hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path, token=token)
        )
        local_hash = sha256_of(local_path)
        remote_hash = sha256_of(downloaded_path)
        status = "OK" if local_hash == remote_hash else "MISMATCH"
        print(f"    {status}: local={local_hash[:12]} remote={remote_hash[:12]}")
        if local_hash != remote_hash:
            mismatches.append(remote_path)

    if mismatches:
        raise SystemExit(f"Verification FAILED for {len(mismatches)} file(s): {mismatches}")
    print("\nVerification passed: all sampled files match byte-for-byte.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload completed run artifacts directly from train/ to Hugging Face Hub.")
    parser.add_argument("--list", action="store_true", help="List what would be uploaded, no network calls.")
    parser.add_argument("--verify", action="store_true", help="Verify an already-completed upload instead of uploading.")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of files to spot-check during --verify.")
    args = parser.parse_args()

    if args.list:
        list_runs()
    elif args.verify:
        verify(sample_size=args.sample_size)
    else:
        upload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
