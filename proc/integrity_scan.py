"""Integrity scan producing a machine-readable manifest (FIXTURE_PLAN.md T3).

Reuses data/dataset_run.py's validation logic (validate_image/validate_video,
run via run_validation()) rather than duplicating it -- that module already
works and is documented as an active tool in books/03_data_cleaning_preprocessing.md,
so this wraps it instead of replacing it. The one thing the existing tools
never did is persist a manifest: run_validation() returns an in-memory list,
data/dataset_fix.py consumes it directly, and nothing survives the process.
This script adds that missing manifest, in the schema proc/manifest.py
defines, so data/dataloader.py can optionally use it to skip known-bad files
(see DatasetBuilder's manifest_path parameter).

Usage:
  python -m proc.integrity_scan --root datasets --out proc/integrity_manifest.json
  python -m proc.integrity_scan --dtype video --limit 500 --out /tmp/quick_check.json
"""
from __future__ import annotations

import argparse

from data.dataset_run import run_validation
from data.dataloader import DatasetBuilder
from proc.manifest import FileRecord, IntegrityManifest, save_manifest


def build_manifest(root="datasets", dtype=None, limit=None, num_workers=8, fail_fast=False) -> IntegrityManifest:
    """Runs the existing raw-readability validator and captures BOTH the
    failures (from run_validation's return value) and the full valid set
    (recomputed from the same DatasetBuilder discovery, since run_validation
    only returns failures today -- not changing that function's contract,
    just deriving the valid set as "everything checked minus the failures")."""
    builder = DatasetBuilder(root=root)
    builder.build()
    samples = builder.samples
    if dtype:
        samples = [s for s in samples if s[2] == dtype]
    else:
        samples = [s for s in samples if s[2] in {"image", "video"}]
    if limit is not None:
        samples = samples[:limit]

    failures = run_validation(root=root, dtype=dtype, limit=limit, fail_fast=fail_fast, num_workers=num_workers)
    failed_paths = {path for path, *_ in failures}
    failure_by_path = {path: (dtype_, error) for path, _label, dtype_, error in failures}

    dataset_by_path = {}
    for record in builder.records:
        dataset_by_path[record["path"]] = record["dataset"]

    files = []
    for path, label, sample_dtype in samples:
        if path in failed_paths:
            _, error = failure_by_path[path]
            files.append(FileRecord(
                path=path, dtype=sample_dtype, dataset=dataset_by_path.get(path, "unknown"),
                label=label, valid=False, error=error,
            ))
        else:
            files.append(FileRecord(
                path=path, dtype=sample_dtype, dataset=dataset_by_path.get(path, "unknown"),
                label=label, valid=True, error=None,
            ))

    manifest = IntegrityManifest(
        root=str(root),
        dtype_filter=dtype,
        total_checked=len(files),
        total_valid=sum(1 for f in files if f.valid),
        total_invalid=sum(1 for f in files if not f.valid),
        files=files,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a machine-readable integrity manifest.")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--dtype", choices=["image", "video"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--out", default="proc/integrity_manifest.json")
    args = parser.parse_args()

    manifest = build_manifest(
        root=args.root, dtype=args.dtype, limit=args.limit,
        num_workers=args.num_workers, fail_fast=args.fail_fast,
    )
    out_path = save_manifest(manifest, args.out)
    print(f"\nManifest written: {out_path}")
    print(f"total_checked={manifest.total_checked} valid={manifest.total_valid} invalid={manifest.total_invalid}")


if __name__ == "__main__":
    main()
