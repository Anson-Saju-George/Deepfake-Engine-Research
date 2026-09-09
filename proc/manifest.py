"""Machine-readable integrity manifest schema (FIXTURE_PLAN.md T3).

The manifest records, per file, whether it passed raw readability validation
(the same check data/dataset_run.py already performs -- open + decode-probe).
It exists so:

  1. proc/integrity_scan.py can persist validation results instead
     of only printing them (the previous behavior: run_validation() returned
     an in-memory list and the CLI printed it -- nothing survived the process).
  2. data/dataloader.py can optionally filter known-bad files out of the
     corpus at build() time, rather than relying solely on the runtime
     decode-failure policy (T1.7) to catch them mid-training. This is the
     "drop" decode-failure policy option that T1.7 deliberately deferred to
     this manifest, rather than implementing a partial version in the data
     layer alone -- see data/dataloader.py:DECODE_FAILURE_POLICIES comment.

Schema is intentionally flat and stdlib-only (json), so nothing beyond the
existing dependency surface is required to read or write it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1


@dataclass
class FileRecord:
    path: str
    dtype: str              # "image" | "video"
    dataset: str
    label: int
    valid: bool
    error: str | None = None


@dataclass
class IntegrityManifest:
    schema_version: int = MANIFEST_SCHEMA_VERSION
    generated_at: float = field(default_factory=time.time)
    root: str = "datasets"
    dtype_filter: str | None = None
    total_checked: int = 0
    total_valid: int = 0
    total_invalid: int = 0
    files: list[FileRecord] = field(default_factory=list)

    def invalid_paths(self) -> set[str]:
        return {f.path for f in self.files if not f.valid}

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "IntegrityManifest":
        files = [FileRecord(**f) for f in d.get("files", [])]
        return cls(
            schema_version=d.get("schema_version", MANIFEST_SCHEMA_VERSION),
            generated_at=d.get("generated_at", 0.0),
            root=d.get("root", "datasets"),
            dtype_filter=d.get("dtype_filter"),
            total_checked=d.get("total_checked", len(files)),
            total_valid=d.get("total_valid", sum(1 for f in files if f.valid)),
            total_invalid=d.get("total_invalid", sum(1 for f in files if not f.valid)),
            files=files,
        )


def save_manifest(manifest: IntegrityManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> IntegrityManifest:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest at {path} has schema_version={data.get('schema_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION}. Regenerate it with the current "
            f"proc/integrity_scan.py before use."
        )
    return IntegrityManifest.from_dict(data)
