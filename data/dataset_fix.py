"""Dataset cleanup utility for unreadable images or videos.

This script reuses the validation logic from data.dataset_run and then applies a
chosen action to unreadable files: report, delete, or quarantine.
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from data.dataset_run import run_validation
except ImportError:
    from dataset_run import run_validation


def quarantine_path(root, quarantine_dir, path):
    root = Path(root).resolve()
    quarantine_dir = Path(quarantine_dir).resolve()
    source = Path(path).resolve()
    relative = source.relative_to(root)
    target = quarantine_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return target


def apply_fix(failures, root, action, quarantine_dir=None):
    print("\n=== FIX ACTION ===")
    print(f"Action: {action}")
    print(f"Failure count: {len(failures)}")

    if not failures:
        print("Nothing to fix.")
        return

    deleted = 0
    moved = 0
    failed = 0

    for path, label, sample_dtype, error in failures:
        try:
            if action == "delete":
                os.remove(path)
                deleted += 1
            elif action == "quarantine":
                quarantine_path(root, quarantine_dir, path)
                moved += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED | {path}")
            print(f"  reason: {type(exc).__name__}: {exc}")

    print("\n=== FIX SUMMARY ===")
    print(f"Deleted    : {deleted}")
    print(f"Quarantined: {moved}")
    print(f"Failed ops : {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate the dataset and delete or quarantine unreadable files."
    )
    parser.add_argument("--root", default="datasets", help="Dataset root path.")
    parser.add_argument("--dtype", choices=["image", "video"], default=None, help="Optional dtype filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick checks.")
    parser.add_argument("--num-workers", type=int, default=8, help="Thread count for validation.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop validation after the first failure.")
    parser.add_argument(
        "--action",
        choices=["report", "delete", "quarantine"],
        default="report",
        help="What to do with unreadable files after validation.",
    )
    parser.add_argument(
        "--quarantine-dir",
        default="datasets_quarantine",
        help="Target directory for quarantined files.",
    )
    args = parser.parse_args()

    failures = run_validation(
        root=args.root,
        dtype=args.dtype,
        limit=args.limit,
        fail_fast=args.fail_fast,
        num_workers=args.num_workers,
    )

    if args.action == "report":
        print("\nReport mode only. No files were modified.")
        return

    apply_fix(
        failures=failures,
        root=args.root,
        action=args.action,
        quarantine_dir=args.quarantine_dir,
    )


if __name__ == "__main__":
    main()
