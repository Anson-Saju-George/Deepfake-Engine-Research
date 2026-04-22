#!/usr/bin/env python3
"""
Interactive model packer / unpacker for completed image and video runs.

Default behavior is menu-driven. Optional subcommands are available for
automation and quick verification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


PACKAGE_MANIFEST_NAME = "weights_manifest.json"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
SCRIPT_PATH = Path(__file__).resolve()
if (
    SCRIPT_PATH.parent.name.lower() == "weights"
    or (SCRIPT_PATH.parent / "Image").is_dir()
    or (SCRIPT_PATH.parent / "Video").is_dir()
    or (SCRIPT_PATH.parent / PACKAGE_MANIFEST_NAME).exists()
):
    REPO_ROOT = SCRIPT_PATH.parent
    EXPORT_ROOT = SCRIPT_PATH.parent
else:
    REPO_ROOT = SCRIPT_PATH.parent
    EXPORT_ROOT = REPO_ROOT / "Weights"
TRAIN_ROOT = REPO_ROOT / "train"
WEIGHTS_ARCHIVE_PATH = REPO_ROOT / "Weights.zip"
SCRIPT_NAME = "model_packer.py"


def sanitize_component(value: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or fallback


def resolve_repo_path(value: str | None, fallback: Path | None = None) -> Path | None:
    if not value:
        return fallback
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "na"


def prompt_text(message: str, default: str | None = None, allow_blank: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{message}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if allow_blank:
            return ""
        print("Value required.")


def prompt_choice(message: str, options: list[str], default_index: int = 0) -> str:
    print(message)
    for index, option in enumerate(options, start=1):
        default_tag = " (default)" if index - 1 == default_index else ""
        print(f"  {index}. {option}{default_tag}")
    while True:
        raw = input("Select option: ").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid selection.")


def select_from_list(title: str, options: list[str]) -> str | None:
    if not options:
        return None
    print(title)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        raw = input("Select option (blank to cancel): ").strip()
        if not raw:
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid selection.")


@dataclass
class BundleSpec:
    domain: str
    family: str
    model_name: str
    experiment_no: str
    checkpoint_path: Path
    summary_payload: dict[str, Any]
    config_payload: dict[str, Any]
    source_dir: Path
    source_kind: str
    checkpoint_suffix: str = ".pth"

    @property
    def run_name_tag(self) -> str:
        run_name = (
            self.summary_payload.get("run_name")
            or self.config_payload.get("run_name")
            or self.source_dir.name
        )
        return sanitize_component(str(run_name))

    @property
    def domain_dir(self) -> str:
        return "Image" if self.domain.lower() == "image" else "Video"

    @property
    def family_model_dir(self) -> str:
        return sanitize_component(f"{self.family}_{self.model_name}")

    @property
    def acc_value(self) -> Any:
        metrics = self.summary_payload.get("test_metrics", {})
        return metrics.get("acc", self.summary_payload.get("best_val_acc"))

    @property
    def f1_value(self) -> Any:
        metrics = self.summary_payload.get("test_metrics", {})
        return metrics.get("f1", self.summary_payload.get("best_val_f1"))

    @property
    def acc_tag(self) -> str:
        return format_metric(self.acc_value)

    @property
    def f1_tag(self) -> str:
        return format_metric(self.f1_value)

    @property
    def checkpoint_name(self) -> str:
        suffix = self.checkpoint_suffix if self.checkpoint_suffix.startswith(".") else f".{self.checkpoint_suffix}"
        return f"best_{self.run_name_tag}_acc-{self.acc_tag}_f1-{self.f1_tag}{suffix}"

    @property
    def summary_name(self) -> str:
        return f"final_summary_{self.run_name_tag}.json"

    @property
    def config_name(self) -> str:
        return f"config_{self.run_name_tag}.json"

    @property
    def archive_name(self) -> str:
        return f"{self.run_name_tag}_acc-{self.acc_tag}_f1-{self.f1_tag}.zip"

    @property
    def general_extract_root(self) -> Path:
        return Path(self.domain.lower()) / self.family_model_dir

    @property
    def archive_path(self) -> Path:
        return EXPORT_ROOT / self.domain_dir / self.family_model_dir / self.archive_name

    def describe(self) -> str:
        return (
            f"{self.experiment_no} | {self.domain_dir} | {self.family} | "
            f"{self.model_name} | acc={self.acc_tag} | f1={self.f1_tag}"
        )


def build_bundle_spec_from_train_run(summary_path: Path) -> BundleSpec:
    summary_payload = load_json(summary_path)
    config_path = summary_path.with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json next to {summary_path}")

    config_payload = load_json(config_path)
    checkpoint_path = resolve_repo_path(summary_payload.get("best_checkpoint"), summary_path.parent / "best.pth")
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint for {summary_path}")

    parts = summary_path.parts
    try:
        train_index = parts.index("train")
        domain = parts[train_index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Could not infer domain from {summary_path}") from exc

    family = str(config_payload.get("family") or summary_path.parent.parent.name)
    model_name = str(config_payload.get("model_name") or summary_payload.get("model_name") or summary_path.parent.name)
    experiment_no = str(summary_payload.get("experiment_no") or config_payload.get("experiment_no") or summary_path.parent.name)

    return BundleSpec(
        domain=domain,
        family=family,
        model_name=model_name,
        experiment_no=experiment_no,
        checkpoint_path=checkpoint_path,
        summary_payload=summary_payload,
        config_payload=config_payload,
        source_dir=summary_path.parent,
        source_kind="train",
        checkpoint_suffix=checkpoint_path.suffix or ".pth",
    )


def discover_train_runs() -> list[BundleSpec]:
    specs: list[BundleSpec] = []
    for summary_path in sorted(TRAIN_ROOT.rglob("final_summary.json")):
        try:
            specs.append(build_bundle_spec_from_train_run(summary_path))
        except Exception as exc:
            print(f"[skip] {summary_path}: {exc}")
    return specs


def choose_existing_file(candidates: list[Path], label: str) -> Path | None:
    options = [str(path) for path in candidates]
    selected = select_from_list(f"Select {label}:", options)
    return Path(selected) if selected else None


def synthesize_generic_summary(
    experiment_no: str,
    model_name: str,
    domain: str,
    checkpoint_path: Path,
    acc_value: str,
    f1_value: str,
    source_dir: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_no": experiment_no,
        "model_name": model_name,
        "source_kind": "generic_directory",
        "source_dir": str(source_dir),
        "best_checkpoint": str(checkpoint_path),
    }
    if domain.lower() == "video":
        payload["category"] = "generic"
    if acc_value or f1_value:
        payload["test_metrics"] = {}
        if acc_value:
            payload["test_metrics"]["acc"] = float(acc_value)
        if f1_value:
            payload["test_metrics"]["f1"] = float(f1_value)
    return payload


def synthesize_generic_config(
    experiment_no: str,
    family: str,
    model_name: str,
    domain: str,
    source_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment_no": experiment_no,
        "family": family,
        "model_name": model_name,
        "domain": domain.lower(),
        "source_kind": "generic_directory",
        "source_dir": str(source_dir),
    }


def build_bundle_spec_from_generic_dir(source_dir: Path) -> BundleSpec:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    checkpoint_candidates = sorted(source_dir.rglob("*.pth"))
    if not checkpoint_candidates:
        raise FileNotFoundError("No .pth checkpoint files found in that directory.")
    checkpoint_path = choose_existing_file(checkpoint_candidates, "checkpoint file")
    if checkpoint_path is None:
        raise RuntimeError("Checkpoint selection cancelled.")

    summary_candidates = sorted(source_dir.rglob("final_summary.json"))
    config_candidates = sorted(source_dir.rglob("config.json"))
    summary_path = choose_existing_file(summary_candidates, "final summary file") if summary_candidates else None
    config_path = choose_existing_file(config_candidates, "config file") if config_candidates else None

    summary_payload = load_json(summary_path) if summary_path else {}
    config_payload = load_json(config_path) if config_path else {}

    domain_default = "image" if "image" in str(source_dir).lower() else "video" if "video" in str(source_dir).lower() else None
    domain = prompt_choice("Choose model domain:", ["image", "video"], default_index=0 if domain_default != "video" else 1)

    family = str(config_payload.get("family") or prompt_text("Family name", default=source_dir.parent.name))
    model_name = str(
        config_payload.get("model_name")
        or summary_payload.get("model_name")
        or prompt_text("Model name", default=sanitize_component(checkpoint_path.stem))
    )
    experiment_no = str(
        summary_payload.get("experiment_no")
        or config_payload.get("experiment_no")
        or prompt_text("Experiment / bundle id", default=sanitize_component(source_dir.name.upper()))
    )

    if not summary_payload:
        acc_value = prompt_text("Test accuracy (blank if unknown)", allow_blank=True)
        f1_value = prompt_text("Test F1 (blank if unknown)", allow_blank=True)
        summary_payload = synthesize_generic_summary(
            experiment_no=experiment_no,
            model_name=model_name,
            domain=domain,
            checkpoint_path=checkpoint_path,
            acc_value=acc_value,
            f1_value=f1_value,
            source_dir=source_dir,
        )

    if not config_payload:
        config_payload = synthesize_generic_config(
            experiment_no=experiment_no,
            family=family,
            model_name=model_name,
            domain=domain,
            source_dir=source_dir,
        )

    return BundleSpec(
        domain=domain,
        family=family,
        model_name=model_name,
        experiment_no=experiment_no,
        checkpoint_path=checkpoint_path,
        summary_payload=summary_payload,
        config_payload=config_payload,
        source_dir=source_dir,
        source_kind="generic",
        checkpoint_suffix=checkpoint_path.suffix or ".pth",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_source_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*") if path.is_file())


def normalize_archive_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def relative_member(path: Path | None, source_dir: Path, source_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return normalize_archive_path(source_root / path.relative_to(source_dir))
    except ValueError:
        return None


def build_source_archive_root(spec: BundleSpec) -> Path:
    if spec.source_kind == "train":
        save_dir = spec.config_payload.get("save_dir") or spec.summary_payload.get("save_dir")
        resolved = resolve_repo_path(str(save_dir)) if save_dir else spec.source_dir
        if resolved is None:
            resolved = spec.source_dir
        try:
            relative = resolved.relative_to(REPO_ROOT)
            return relative
        except ValueError:
            pass
    return Path("source_bundle") / sanitize_component(spec.source_dir.name)


def ensure_export_root() -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_export_script_copy() -> Path:
    ensure_export_root()
    target = EXPORT_ROOT / SCRIPT_NAME
    if SCRIPT_PATH != target:
        target.write_bytes(SCRIPT_PATH.read_bytes())
    return target


def extract_primary_metrics(summary_payload: dict[str, Any], evaluation_payload: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    evaluation_metrics = (evaluation_payload or {}).get("metrics", {})
    summary_metrics = summary_payload.get("test_metrics", {})

    for key in ("loss", "acc", "f1", "precision", "recall", "roc_auc", "average_precision"):
        value = evaluation_metrics.get(key, summary_metrics.get(key))
        if value is not None:
            metrics[key] = value
    return metrics


def build_compact_metrics_payload(
    spec: BundleSpec,
    summary_payload: dict[str, Any],
    config_payload: dict[str, Any],
    evaluation_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "experiment_no": spec.experiment_no,
        "run_name": spec.run_name_tag,
        "domain": spec.domain.lower(),
        "family": spec.family,
        "model_name": spec.model_name,
        "source_kind": spec.source_kind,
        "metrics": extract_primary_metrics(summary_payload, evaluation_payload),
        "paths": {
            "checkpoint": spec.checkpoint_name,
            "config": spec.config_name,
            "final_summary": spec.summary_name,
            "test_evaluation": f"test_evaluation_{spec.run_name_tag}.json" if evaluation_payload else None,
        },
        "config_summary": {
            "dataset_scope": summary_payload.get("dataset_scope"),
            "dataset_tag": summary_payload.get("dataset_tag"),
            "category": summary_payload.get("category") or config_payload.get("category"),
            "mode": config_payload.get("mode"),
            "seq_len": config_payload.get("seq_len"),
        },
    }


def build_bundle_manifest(spec: BundleSpec, source_root: Path) -> dict[str, Any]:
    config_path = spec.source_dir / "config.json"
    final_summary_path = spec.source_dir / "final_summary.json"
    evaluation_path = spec.source_dir / "test_evaluation.json"
    predictions_path = spec.source_dir / "test_predictions.csv"
    history_path = spec.source_dir / "history.csv"
    run_record_path = spec.source_dir / "run_record.md"
    split_summary_path = spec.source_dir / "split_summary.json"
    best_summary_path = spec.source_dir / "best_summary.json"

    return {
        "format_version": 2,
        "bundle_id": spec.run_name_tag,
        "archive_name": spec.archive_name,
        "experiment_no": spec.experiment_no,
        "domain": spec.domain.lower(),
        "family": spec.family,
        "model_name": spec.model_name,
        "source_kind": spec.source_kind,
        "family_model_dir": spec.family_model_dir,
        "general_extract_root": normalize_archive_path(spec.general_extract_root),
        "restore_root": normalize_archive_path(source_root),
        "members": {
            "checkpoint": relative_member(spec.checkpoint_path, spec.source_dir, source_root),
            "config": (
                relative_member(config_path, spec.source_dir, source_root)
                if config_path.exists()
                else "metadata/config.json"
            ),
            "final_summary": (
                relative_member(final_summary_path, spec.source_dir, source_root)
                if final_summary_path.exists()
                else "metadata/final_summary.json"
            ),
            "test_evaluation": relative_member(evaluation_path, spec.source_dir, source_root) if evaluation_path.exists() else None,
            "test_predictions": relative_member(predictions_path, spec.source_dir, source_root) if predictions_path.exists() else None,
            "history_csv": relative_member(history_path, spec.source_dir, source_root) if history_path.exists() else None,
            "run_record": relative_member(run_record_path, spec.source_dir, source_root) if run_record_path.exists() else None,
            "split_summary": relative_member(split_summary_path, spec.source_dir, source_root) if split_summary_path.exists() else None,
            "best_summary": relative_member(best_summary_path, spec.source_dir, source_root) if best_summary_path.exists() else None,
        },
        "general_files": {
            "checkpoint_name": spec.checkpoint_name,
            "config_name": spec.config_name,
            "summary_name": spec.summary_name,
            "evaluation_name": f"test_evaluation_{spec.run_name_tag}.json",
            "metrics_name": f"metrics_{spec.run_name_tag}.json",
        },
    }


def read_bundle_manifest(zip_path: Path) -> dict[str, Any] | None:
    with ZipFile(zip_path, "r") as archive:
        if BUNDLE_MANIFEST_NAME not in archive.namelist():
            return None
        return json.loads(archive.read(BUNDLE_MANIFEST_NAME).decode("utf-8"))


def create_weights_archive() -> Path:
    ensure_export_script_copy()
    manifest = build_export_manifest()
    compression_kwargs = {"compression": ZIP_DEFLATED}
    if sys.version_info >= (3, 7):
        compression_kwargs["compresslevel"] = 9

    with ZipFile(WEIGHTS_ARCHIVE_PATH, mode="w", **compression_kwargs) as archive:
        archive.writestr(PACKAGE_MANIFEST_NAME, write_json_bytes(manifest))
        archive.writestr(SCRIPT_NAME, SCRIPT_PATH.read_bytes())
        for file_path in sorted(path for path in EXPORT_ROOT.rglob("*") if path.is_file()):
            if file_path == EXPORT_ROOT / SCRIPT_NAME:
                continue
            if file_path.name == PACKAGE_MANIFEST_NAME:
                continue
            archive.write(file_path, arcname=str(file_path.relative_to(EXPORT_ROOT)))
    return WEIGHTS_ARCHIVE_PATH


def create_archive(spec: BundleSpec) -> Path:
    ensure_export_script_copy()
    ensure_parent(spec.archive_path)
    source_root = build_source_archive_root(spec)
    bundle_manifest = build_bundle_manifest(spec, source_root)
    compression_kwargs = {"compression": ZIP_DEFLATED}
    if sys.version_info >= (3, 7):
        compression_kwargs["compresslevel"] = 9

    with ZipFile(spec.archive_path, mode="w", **compression_kwargs) as archive:
        archive.writestr(BUNDLE_MANIFEST_NAME, write_json_bytes(bundle_manifest))
        if not (spec.source_dir / "config.json").exists():
            archive.writestr(bundle_manifest["members"]["config"], write_json_bytes(spec.config_payload))
        if not (spec.source_dir / "final_summary.json").exists():
            archive.writestr(bundle_manifest["members"]["final_summary"], write_json_bytes(spec.summary_payload))
        for file_path in iter_source_files(spec.source_dir):
            relative = file_path.relative_to(spec.source_dir)
            archive.write(file_path, arcname=str(source_root / relative))

    return spec.archive_path


def discover_archives() -> list[Path]:
    if not EXPORT_ROOT.exists():
        return []
    return sorted(
        path for path in EXPORT_ROOT.rglob("*.zip")
        if path.resolve() != WEIGHTS_ARCHIVE_PATH.resolve()
    )


def build_archive_record(zip_path: Path) -> dict[str, Any]:
    bundle_manifest = read_bundle_manifest(zip_path)
    if bundle_manifest is not None:
        return {
            "archive": normalize_archive_path(zip_path.relative_to(EXPORT_ROOT)),
            "experiment_no": bundle_manifest.get("experiment_no"),
            "bundle_id": bundle_manifest.get("bundle_id"),
            "domain": bundle_manifest.get("domain"),
            "family": bundle_manifest.get("family"),
            "model_name": bundle_manifest.get("model_name"),
            "family_model_dir": bundle_manifest.get("family_model_dir"),
            "general_extract_root": bundle_manifest.get("general_extract_root"),
            "format_version": bundle_manifest.get("format_version", 2),
        }

    summary_payload, config_payload, bundle_root = read_bundle_payloads(zip_path)
    return {
        "archive": normalize_archive_path(zip_path.relative_to(EXPORT_ROOT)),
        "experiment_no": summary_payload.get("experiment_no") or config_payload.get("experiment_no"),
        "bundle_id": summary_payload.get("run_name") or config_payload.get("run_name") or zip_path.stem,
        "domain": config_payload.get("domain") or ("image" if "Image" in bundle_root else "video"),
        "family": config_payload.get("family"),
        "model_name": config_payload.get("model_name") or summary_payload.get("model_name"),
        "family_model_dir": Path(bundle_root).name,
        "general_extract_root": normalize_archive_path(Path((config_payload.get("domain") or "image").lower()) / Path(bundle_root).name),
        "format_version": 1,
    }


def build_export_manifest() -> dict[str, Any]:
    archives = discover_archives()
    records = [build_archive_record(path) for path in archives]
    return {
        "format_version": 1,
        "package_type": "weights_bundle",
        "bundle_count": len(records),
        "archives": records,
    }


def read_bundle_payloads(zip_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    with ZipFile(zip_path, "r") as archive:
        if BUNDLE_MANIFEST_NAME in archive.namelist():
            bundle_manifest = json.loads(archive.read(BUNDLE_MANIFEST_NAME).decode("utf-8"))
            members = bundle_manifest.get("members", {})
            summary_name = members.get("final_summary")
            config_name = members.get("config")
            if summary_name is None or config_name is None:
                raise RuntimeError(f"Archive is missing manifest summary/config members: {zip_path}")
            summary_payload = json.loads(archive.read(summary_name).decode("utf-8"))
            config_payload = json.loads(archive.read(config_name).decode("utf-8"))
            bundle_root = bundle_manifest.get("restore_root") or str(Path(summary_name).parent)
            return summary_payload, config_payload, bundle_root

        names = archive.namelist()
        summary_name = next((name for name in names if name.endswith(".json") and "/final_summary_" in f"/{name}"), None)
        config_name = next((name for name in names if name.endswith(".json") and "/config_" in f"/{name}"), None)
        if summary_name is None or config_name is None:
            raise RuntimeError(f"Archive is missing summary/config payloads: {zip_path}")
        bundle_root = str(Path(summary_name).parent)
        summary_payload = json.loads(archive.read(summary_name).decode("utf-8"))
        config_payload = json.loads(archive.read(config_name).decode("utf-8"))
    return summary_payload, config_payload, bundle_root


def archive_contains_original_train_tree(zip_path: Path, restore_dir: Path) -> bool:
    restore_prefix = normalize_archive_path(restore_dir.relative_to(REPO_ROOT))
    with ZipFile(zip_path, "r") as archive:
        return any(name.startswith(f"{restore_prefix}/") for name in archive.namelist())


def read_json_member(archive: ZipFile, member: str | None) -> dict[str, Any] | None:
    if not member:
        return None
    try:
        return json.loads(archive.read(member).decode("utf-8"))
    except KeyError:
        return None


def build_general_unpack_plan(zip_path: Path) -> tuple[Path, dict[str, bytes]]:
    bundle_manifest = read_bundle_manifest(zip_path)
    with ZipFile(zip_path, "r") as archive:
        if bundle_manifest is not None:
            members = bundle_manifest.get("members", {})
            file_names = bundle_manifest.get("general_files", {})
            output_root = Path(bundle_manifest["general_extract_root"])
            summary_payload = read_json_member(archive, members.get("final_summary")) or {}
            config_payload = read_json_member(archive, members.get("config")) or {}
            evaluation_payload = read_json_member(archive, members.get("test_evaluation"))

            compact_payload = {
                "experiment_no": bundle_manifest.get("experiment_no"),
                "run_name": bundle_manifest.get("bundle_id"),
                "domain": bundle_manifest.get("domain"),
                "family": bundle_manifest.get("family"),
                "model_name": bundle_manifest.get("model_name"),
                "metrics": extract_primary_metrics(summary_payload, evaluation_payload),
            }

            outputs: dict[str, bytes] = {}
            checkpoint_member = members.get("checkpoint")
            if checkpoint_member:
                outputs[file_names["checkpoint_name"]] = archive.read(checkpoint_member)
            if summary_payload:
                outputs[file_names["summary_name"]] = write_json_bytes(summary_payload)
            if config_payload:
                outputs[file_names["config_name"]] = write_json_bytes(config_payload)
            if evaluation_payload:
                outputs[file_names["evaluation_name"]] = write_json_bytes(evaluation_payload)
            outputs[file_names["metrics_name"]] = write_json_bytes(compact_payload)
            return output_root, outputs

        summary_payload, config_payload, bundle_root = read_bundle_payloads(zip_path)
        metrics_payload = {
            "experiment_no": summary_payload.get("experiment_no") or config_payload.get("experiment_no"),
            "run_name": summary_payload.get("run_name") or config_payload.get("run_name") or zip_path.stem,
            "domain": (config_payload.get("domain") or ("image" if "Image" in bundle_root else "video")).lower(),
            "family": config_payload.get("family"),
            "model_name": config_payload.get("model_name") or summary_payload.get("model_name"),
            "metrics": extract_primary_metrics(summary_payload, None),
        }
        output_root = Path(metrics_payload["domain"]) / Path(bundle_root).name
        outputs = {
            next(name for name in archive.namelist() if name.endswith(".pth") and "/Weights/" in f"/{name}").split("/")[-1]: archive.read(
                next(name for name in archive.namelist() if name.endswith(".pth") and "/Weights/" in f"/{name}")
            ),
            Path(next(name for name in archive.namelist() if name.endswith(".json") and "/final_summary_" in f"/{name}")).name: write_json_bytes(summary_payload),
            Path(next(name for name in archive.namelist() if name.endswith(".json") and "/config_" in f"/{name}")).name: write_json_bytes(config_payload),
            f"metrics_{sanitize_component(str(metrics_payload['run_name']))}.json": write_json_bytes(metrics_payload),
        }
        return output_root, outputs


def unpack_archive(zip_path: Path, destination_root: Path) -> list[Path]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    destination_root.mkdir(parents=True, exist_ok=True)
    target_root, output_map = build_general_unpack_plan(zip_path)
    extracted: list[Path] = []
    for relative_name, payload in output_map.items():
        target_path = destination_root / target_root / relative_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(payload)
        extracted.append(target_path)
    return extracted


def unpack_all_archives(destination_root: Path) -> list[Path]:
    archives = discover_archives()
    if not archives:
        raise RuntimeError("No archives found under Weights.")
    extracted: list[Path] = []
    for archive_path in archives:
        extracted.extend(unpack_archive(archive_path, destination_root))
    return extracted


def infer_train_restore_dir(summary_payload: dict[str, Any], config_payload: dict[str, Any]) -> Path:
    save_dir_value = config_payload.get("save_dir") or summary_payload.get("save_dir")
    if not save_dir_value:
        raise RuntimeError("Archive does not contain an original train save_dir.")
    restore_dir = resolve_repo_path(str(save_dir_value))
    if restore_dir is None:
        raise RuntimeError("Could not resolve train save_dir from archive.")
    try:
        restore_dir.relative_to(TRAIN_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to restore outside train/: {restore_dir}") from exc
    return restore_dir


def restore_archive_to_train(zip_path: Path) -> list[Path]:
    summary_payload, config_payload, _ = read_bundle_payloads(zip_path)
    restore_dir = infer_train_restore_dir(summary_payload, config_payload)
    restore_dir.mkdir(parents=True, exist_ok=True)

    if archive_contains_original_train_tree(zip_path, restore_dir):
        written_paths: list[Path] = []
        restore_prefix = normalize_archive_path(restore_dir.relative_to(REPO_ROOT))
        with ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                if not member.startswith(f"{restore_prefix}/") or member.endswith("/"):
                    continue
                target = REPO_ROOT / Path(member)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                written_paths.append(target)
        return written_paths

    written_paths = []
    with ZipFile(zip_path, "r") as archive:
        checkpoint_name = next((name for name in archive.namelist() if name.endswith(".pth")), None)
        if checkpoint_name is None:
            raise RuntimeError(f"Archive is missing checkpoint payload: {zip_path}")

        checkpoint_path = restore_dir / "best.pth"
        checkpoint_path.write_bytes(archive.read(checkpoint_name))
        written_paths.append(checkpoint_path)

    config_path = restore_dir / "config.json"
    config_path.write_bytes(write_json_bytes(config_payload))
    written_paths.append(config_path)

    summary_path = restore_dir / "final_summary.json"
    summary_path.write_bytes(write_json_bytes(summary_payload))
    written_paths.append(summary_path)

    return written_paths


def restore_all_archives_to_train() -> list[Path]:
    archives = discover_archives()
    if not archives:
        raise RuntimeError("No archives found under Weights.")
    restored: list[Path] = []
    for archive_path in archives:
        restored.extend(restore_archive_to_train(archive_path))
    return restored


def print_specs(specs: list[BundleSpec]) -> None:
    if not specs:
        print("No completed train runs found.")
        return
    print(f"Completed train runs found: {len(specs)}")
    for index, spec in enumerate(specs, start=1):
        print(f"{index:>2}. {spec.describe()}")


def find_spec_by_experiment(specs: list[BundleSpec], experiment_no: str) -> BundleSpec | None:
    matches = [spec for spec in specs if spec.experiment_no == experiment_no]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        details = "\n".join(f"  - {match.describe()} | run={match.run_name_tag}" for match in matches)
        raise RuntimeError(
            f"Experiment {experiment_no} matched multiple completed runs.\n"
            f"Use the interactive menu for an exact selection.\n{details}"
        )
    return None


def menu_pack_single_train_run() -> None:
    specs = discover_train_runs()
    if not specs:
        print("No completed train runs available.")
        return
    choices = [spec.describe() for spec in specs]
    selected = select_from_list("Select completed train run to pack:", choices)
    if selected is None:
        return
    selected_index = choices.index(selected)
    archive_path = create_archive(specs[selected_index])
    weights_archive = create_weights_archive()
    print(f"\nPacked: {archive_path}")
    print(f"Updated export bundle: {weights_archive}")


def menu_pack_all_train_runs() -> None:
    specs = discover_train_runs()
    if not specs:
        print("No completed train runs available.")
        return
    for spec in specs:
        archive_path = create_archive(spec)
        print(f"Packed: {archive_path}")
    weights_archive = create_weights_archive()
    print(f"\nPacked {len(specs)} completed runs.")
    print(f"Updated export bundle: {weights_archive}")


def menu_pack_generic_dir() -> None:
    source_dir = Path(prompt_text("Generic model directory path"))
    spec = build_bundle_spec_from_generic_dir(source_dir)
    archive_path = create_archive(spec)
    weights_archive = create_weights_archive()
    print(f"\nPacked: {archive_path}")
    print(f"Updated export bundle: {weights_archive}")


def menu_unpack_archive() -> None:
    zip_candidates = discover_archives()
    selected_zip = choose_existing_file(zip_candidates, "archive") if zip_candidates else None
    if selected_zip is None:
        raw_path = prompt_text("Archive path", allow_blank=True)
        if not raw_path:
            return
        selected_zip = Path(raw_path)
    destination = Path(prompt_text("Destination root", default=str(REPO_ROOT)))
    extracted = unpack_archive(selected_zip, destination)
    print(f"\nUnpacked general-user artifacts to: {destination}")
    for path in extracted:
        print(f"  - {path}")


def menu_unpack_all_archives() -> None:
    destination = Path(prompt_text("Destination root", default=str(REPO_ROOT)))
    extracted = unpack_all_archives(destination)
    print(f"\nUnpacked {len(extracted)} general-user files from all archives into: {destination}")


def menu_restore_archive_to_train() -> None:
    zip_candidates = discover_archives()
    selected_zip = choose_existing_file(zip_candidates, "archive") if zip_candidates else None
    if selected_zip is None:
        raw_path = prompt_text("Archive path", allow_blank=True)
        if not raw_path:
            return
        selected_zip = Path(raw_path)
    restored = restore_archive_to_train(selected_zip)
    print(f"\nRestored researcher train artifacts from: {selected_zip}")
    for path in restored:
        print(f"  - {path}")


def menu_restore_all_archives_to_train() -> None:
    restored = restore_all_archives_to_train()
    print(f"\nRestored {len(restored)} researcher files from all archives back into train/")


def run_interactive_menu() -> int:
    actions = {
        "1": ("List completed train runs", lambda: print_specs(discover_train_runs())),
        "2": ("Pack one completed train run", menu_pack_single_train_run),
        "3": ("Pack all completed train runs", menu_pack_all_train_runs),
        "4": ("Pack a generic model directory", menu_pack_generic_dir),
        "5": ("Unpack a model archive for general users", menu_unpack_archive),
        "6": ("Unpack all model archives for general users", menu_unpack_all_archives),
        "7": ("Restore one archive back to original train dir", menu_restore_archive_to_train),
        "8": ("Restore all archives back to original train dir", menu_restore_all_archives_to_train),
        "0": ("Exit", None),
    }

    while True:
        print("\nMODEL PACKER")
        print("============")
        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        choice = input("Select option: ").strip()
        if choice == "0":
            return 0
        action = actions.get(choice)
        if action is None:
            print("Invalid selection.")
            continue
        try:
            action[1]()
        except Exception as exc:
            print(f"[error] {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack and unpack trained model bundles for general or researcher use.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List completed train runs that can be packed.")

    pack_train = subparsers.add_parser("pack-train", help="Pack completed train runs.")
    pack_train.add_argument("--exp", help="Single experiment number to pack.")
    pack_train.add_argument("--all", action="store_true", help="Pack every completed train run.")

    pack_dir = subparsers.add_parser("pack-dir", help="Pack a generic model directory.")
    pack_dir.add_argument("source_dir", help="Directory containing checkpoint/config/result files.")

    unpack_cmd = subparsers.add_parser("unpack", help="Unpack a created model archive into general-user model directories.")
    unpack_cmd.add_argument("archive", nargs="?", help="Path to a .zip bundle.")
    unpack_cmd.add_argument("--all", action="store_true", help="Unpack all bundles under Weights.")
    unpack_cmd.add_argument("--dest", default=str(REPO_ROOT), help="Destination root for extraction.")

    restore_cmd = subparsers.add_parser("restore-train", help="Restore bundled artifacts back into original train save_dir for researchers.")
    restore_cmd.add_argument("archive", nargs="?", help="Path to a .zip bundle.")
    restore_cmd.add_argument("--all", action="store_true", help="Restore all bundles under Weights back into train/.")

    return parser


def handle_args(args: argparse.Namespace) -> int:
    if not args.command:
        return run_interactive_menu()

    if args.command == "list":
        print_specs(discover_train_runs())
        return 0

    if args.command == "pack-train":
        specs = discover_train_runs()
        if args.all:
            for spec in specs:
                print(create_archive(spec))
            print(create_weights_archive())
            return 0
        if args.exp:
            spec = find_spec_by_experiment(specs, args.exp)
            if spec is None:
                raise SystemExit(f"Experiment not found: {args.exp}")
            print(create_archive(spec))
            print(create_weights_archive())
            return 0
        raise SystemExit("Use --exp <ID> or --all with pack-train.")

    if args.command == "pack-dir":
        spec = build_bundle_spec_from_generic_dir(Path(args.source_dir))
        print(create_archive(spec))
        print(create_weights_archive())
        return 0

    if args.command == "unpack":
        if args.all:
            extracted = unpack_all_archives(Path(args.dest))
            print(f"Extracted {len(extracted)} files from all archives into {Path(args.dest)}")
            return 0
        if not args.archive:
            raise SystemExit("Use unpack <archive> or unpack --all.")
        extracted = unpack_archive(Path(args.archive), Path(args.dest))
        print(f"Extracted {len(extracted)} files into {Path(args.dest)}")
        return 0

    if args.command == "restore-train":
        if args.all:
            restored = restore_all_archives_to_train()
            print(f"Restored {len(restored)} files from all archives back into train/")
            return 0
        if not args.archive:
            raise SystemExit("Use restore-train <archive> or restore-train --all.")
        restored = restore_archive_to_train(Path(args.archive))
        print(f"Restored {len(restored)} files back into the original train directory")
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return handle_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
