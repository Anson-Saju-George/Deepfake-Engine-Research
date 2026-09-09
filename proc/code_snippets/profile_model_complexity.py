from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import timm
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"timm is required for profiling: {exc}") from exc

import sys as _sys
if str(Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
from train.common.temporal_heads import ConvLSTMHead, TemporalConvHead


ROOT = Path(__file__).resolve().parent
TRAIN_ROOTS = [ROOT / "train" / "image", ROOT / "train" / "video"]
OUT_PATH = ROOT / "MODEL_COMPLEXITY.md"
IMAGE_SHAPE = (1, 3, 224, 224)
VIDEO_SHAPE = (1, 4, 3, 224, 224)
WARMUP_RUNS = int(os.environ.get("PROFILE_WARMUP", "8"))
BENCH_RUNS = int(os.environ.get("PROFILE_RUNS", "30"))


class ProfileVideoClassifier(nn.Module):
    """Architecture-only clone of the training video model.

    The trainer uses pretrained=True, but complexity and latency only depend on
    architecture shape. pretrained=False avoids network access and checkpoint mutation.
    """

    def __init__(self, model_name: str, num_classes: int = 2, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise RuntimeError(f"Could not determine feature dimension for model '{model_name}'")
        self.temporal_head = str(config.get("temporal_head", "mean")).lower()
        hidden_dim = int(config.get("temporal_hidden_dim", min(feature_dim, 512)))
        num_layers = int(config.get("temporal_layers", 1))
        dropout = float(config.get("temporal_dropout", 0.2))

        if self.temporal_head == "mean":
            self.head = nn.Linear(feature_dim, num_classes)
        elif self.temporal_head == "lstm":
            self.temporal = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        elif self.temporal_head == "gru":
            self.temporal = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        elif self.temporal_head == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=int(config.get("temporal_attention_heads", 8)),
                dim_feedforward=int(config.get("temporal_ff_dim", hidden_dim * 4)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))
        elif self.temporal_head == "tcn":
            self.temporal = TemporalConvHead(feature_dim, hidden_dim, num_classes, dropout)
        elif self.temporal_head == "convlstm":
            self.temporal = ConvLSTMHead(feature_dim, hidden_dim, num_classes, dropout)
        else:
            raise RuntimeError(f"Unsupported temporal head: {self.temporal_head}")

    def _forward_frames(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        return feats

    def _forward_frame_maps(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "forward_features"):
            raise RuntimeError("ConvLSTM head requires a timm backbone with forward_features().")
        feats = self.backbone.forward_features(x)
        if feats.ndim != 4:
            raise RuntimeError(f"ConvLSTM head expected 4D feature maps, got shape={tuple(feats.shape)}")
        if feats.shape[1] != getattr(self.backbone, "num_features", feats.shape[1]) and feats.shape[-1] == getattr(
            self.backbone, "num_features", feats.shape[-1]
        ):
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.head(self._forward_frames(x))
        if x.ndim != 5:
            raise ValueError(f"Expected 4D or 5D input, got shape={tuple(x.shape)}")
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        if self.temporal_head == "convlstm":
            frame_maps = self._forward_frame_maps(x)
            maps = frame_maps.view(batch_size, seq_len, frame_maps.shape[1], frame_maps.shape[2], frame_maps.shape[3])
            return self.temporal(maps)
        feats = self._forward_frames(x).view(batch_size, seq_len, -1)
        if self.temporal_head == "mean":
            return self.head(feats.mean(dim=1))
        if self.temporal_head in {"lstm", "gru"}:
            temporal_out, _ = self.temporal(feats)
            return self.head(temporal_out[:, -1])
        if self.temporal_head == "transformer":
            temporal_out = self.temporal(feats)
            return self.head(temporal_out.mean(dim=1))
        if self.temporal_head == "tcn":
            return self.temporal(feats)
        raise RuntimeError(f"Unsupported temporal head: {self.temporal_head}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def category_from_path(path: Path, config: dict[str, Any]) -> str:
    if "train/image" in path.as_posix():
        return "Image architectures"
    category = str(config.get("category", "")).lower()
    if category == "spatial":
        return "Spatial video architectures"
    if category == "temporal":
        return "Temporal video architectures"
    if category == "spatiotemporal":
        return "Spatiotemporal video architectures"
    return "Other"


def discover_runs() -> list[dict[str, Any]]:
    runs = []
    for root in TRAIN_ROOTS:
        for final_path in sorted(root.rglob("final_summary.json")):
            run_dir = final_path.parent
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue
            config = load_json(config_path)
            final = load_json(final_path)
            if "train/image" in final_path.as_posix():
                model_kind = "image"
                temporal_head = "image"
                shape = IMAGE_SHAPE
            else:
                model_kind = "video"
                temporal_head = str(config.get("temporal_head", "mean")).lower()
                shape = VIDEO_SHAPE
            runs.append(
                {
                    "run_dir": run_dir,
                    "config": config,
                    "final": final,
                    "model_kind": model_kind,
                    "category": category_from_path(final_path, config),
                    "experiment": final.get("experiment_no") or config.get("experiment_no"),
                    "family": config.get("family") or run_dir.parent.name,
                    "model_name": config.get("model_name") or final.get("model_name"),
                    "temporal_head": temporal_head,
                    "shape": shape,
                    "loss": final.get("loss_mode") or config.get("loss_mode") or "cross_entropy",
                    "scope": final.get("dataset_scope") or config.get("dataset_scope") or config.get("dataset_tag"),
                }
            )
    return runs


def unique_architectures(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in runs:
        key = (
            run["category"],
            run["family"],
            run["model_name"],
            run["model_kind"],
            run["temporal_head"],
            run["shape"],
        )
        if key not in by_key:
            by_key[key] = {**run, "sources": []}
        by_key[key]["sources"].append(f"{run['experiment']}:{run['scope']}:{run['loss']}")
    return list(by_key.values())


def build_model(run: dict[str, Any]) -> nn.Module:
    if run["model_kind"] == "image":
        return timm.create_model(run["model_name"], pretrained=False, num_classes=2)
    return ProfileVideoClassifier(run["model_name"], num_classes=2, config=run["config"])


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def checkpoint_size_mb(run: dict[str, Any]) -> float | None:
    best = run["final"].get("best_checkpoint")
    if not best:
        return None
    path = ROOT / best
    if not path.exists():
        path = run["run_dir"] / "best.pth"
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def profile_flops(model: nn.Module, sample: torch.Tensor) -> tuple[float | None, float | None, str]:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(sample,), verbose=False)
        return float(macs) * 2.0 / 1e9, float(macs) / 1e9, "thop"
    except Exception as thop_exc:
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = FlopCountAnalysis(model, sample).total()
            return float(flops) / 1e9, float(flops) / 2.0 / 1e9, "fvcore"
        except Exception as fvcore_exc:
            return None, None, f"unavailable: thop={type(thop_exc).__name__}: {thop_exc}; fvcore={type(fvcore_exc).__name__}: {fvcore_exc}"


@torch.inference_mode()
def benchmark(model: nn.Module, sample: torch.Tensor, device: torch.device) -> tuple[float | None, float | None, float | None, str]:
    try:
        model = model.to(device).eval()
        sample = sample.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        for _ in range(WARMUP_RUNS):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(BENCH_RUNS):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / BENCH_RUNS) * 1000.0
        throughput = 1000.0 / latency_ms if latency_ms > 0 else None
        memory_mb = None
        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return latency_ms, throughput, memory_mb, ""
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


def profile_one(run: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {
        "group": run["category"],
        "model": f"{run['family']} / {run['model_name']}",
        "head": run["temporal_head"],
        "shape": "x".join(str(x) for x in run["shape"][1:]),
        "sources": ", ".join(run["sources"]),
        "params_m": None,
        "trainable_m": None,
        "flops_g": None,
        "macs_g": None,
        "flops_note": "",
        "latency_ms": None,
        "throughput": None,
        "memory_mb": None,
        "checkpoint_mb": checkpoint_size_mb(run),
        "status": "ok",
    }
    try:
        model = build_model(run).eval()
        total, trainable = count_params(model)
        result["params_m"] = total / 1e6
        result["trainable_m"] = trainable / 1e6
        sample = torch.randn(*run["shape"])
        result["flops_g"], result["macs_g"], result["flops_note"] = profile_flops(model.cpu(), sample)
        latency_ms, throughput, memory_mb, bench_error = benchmark(model, sample, device)
        result["latency_ms"] = latency_ms
        result["throughput"] = throughput
        result["memory_mb"] = memory_mb
        if bench_error:
            result["status"] = bench_error
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result["status"] = f"{type(exc).__name__}: {exc}"
    return result


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Input | Params (M) | Trainable Params (M) | FLOPs (G) | MACs (G) | Latency (ms) | Throughput | Memory (MB) | Checkpoint MB | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        note = row["status"] if row["status"] != "ok" else row["flops_note"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"] + f" ({row['head']})",
                    row["shape"],
                    fmt_num(row["params_m"], 2),
                    fmt_num(row["trainable_m"], 2),
                    fmt_num(row["flops_g"], 2),
                    fmt_num(row["macs_g"], 2),
                    fmt_num(row["latency_ms"], 2),
                    fmt_num(row["throughput"], 2),
                    fmt_num(row["memory_mb"], 2),
                    fmt_num(row["checkpoint_mb"], 2),
                    note.replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(results: list[dict[str, Any]], device: torch.device) -> None:
    ok = [row for row in results if row["status"] == "ok"]
    efficient = min(ok, key=lambda r: (r["latency_ms"] if r["latency_ms"] is not None else float("inf")), default=None)
    largest = max(results, key=lambda r: (r["params_m"] if r["params_m"] is not None else -1), default=None)
    perf_rows = [r for r in results if r["throughput"] and r["params_m"]]
    perf_compute = max(perf_rows, key=lambda r: r["throughput"] / r["params_m"], default=None)

    lines = [
        "# Model Complexity Audit",
        "",
        "Generated by `profile_model_complexity.py` from completed experiment configs only.",
        "",
        "No training was started. Checkpoints were not loaded or modified. Architectures were instantiated with `pretrained=False` to avoid downloads while preserving model structure.",
        "",
        f"Device used: `{device}`",
        f"CUDA device: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}`",
        f"Warmup runs: `{WARMUP_RUNS}`",
        f"Timed runs: `{BENCH_RUNS}`",
        "Image input: `(1, 3, 224, 224)`",
        "Video input: `(1, 4, 3, 224, 224)`",
        "",
        "FLOPs/MACs are computed with `thop` when available, then `fvcore` as fallback. If neither can profile a model safely, FLOPs and MACs are reported as `NA` with the profiler error in Notes.",
        "",
    ]
    for group in ["Image architectures", "Spatial video architectures", "Temporal video architectures", "Spatiotemporal video architectures"]:
        group_rows = [row for row in results if row["group"] == group]
        if not group_rows:
            continue
        lines.extend([f"## {group}", "", markdown_table(group_rows), ""])

    lines.extend(["## Profiling Failures / Limitations", ""])
    failures = [row for row in results if row["status"] != "ok" or row["flops_g"] is None]
    if failures:
        for row in failures:
            lines.append(f"- `{row['model']} ({row['head']})`: {row['status'] if row['status'] != 'ok' else row['flops_note']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Short Summary", ""])
    if efficient:
        lines.append(f"- Most computationally efficient by latency: `{efficient['model']} ({efficient['head']})` at `{fmt_num(efficient['latency_ms'], 2)} ms` per sample.")
    if largest:
        lines.append(f"- Largest model by parameters: `{largest['model']} ({largest['head']})` with `{fmt_num(largest['params_m'], 2)}M` parameters.")
    if perf_compute:
        lines.append(
            f"- Best performance-per-compute proxy by throughput per million parameters: `{perf_compute['model']} ({perf_compute['head']})`."
        )
    lines.append("")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = discover_runs()
    archs = unique_architectures(runs)
    results = []
    for idx, run in enumerate(archs, 1):
        print(f"[{idx}/{len(archs)}] profiling {run['category']} | {run['family']} | {run['model_name']} | {run['temporal_head']}", flush=True)
        results.append(profile_one(run, device))
    write_report(results, device)
    print(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
