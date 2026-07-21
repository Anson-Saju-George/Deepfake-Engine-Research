"""Shared real video trainer for image-style video backbones."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataloader import DatasetBuilder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42


VIDEO_DATASET_SCOPE_CHOICES = [
    "celebdf",
    "ffpp",
    "video_combined",
    "real_ai_videos",
    "video_all",
]

DEFAULT_VIDEO_SMOKE_RUNTIME = {
    "decode_backend": "cv2",
    "transform_profile": "default",
    "gpu_aug_mode": "none",
    "image_size": 224,
    "workers": 8,
    "prefetch_factor": 4,
    "reader_cache": 32,
}


class LogFileStream:
    def __init__(self, stream) -> None:
        self.stream = stream
        self._line_buffer = ""

    def write(self, data):
        text = str(data)
        parts = text.split("\r")
        for index, chunk in enumerate(parts):
            if index > 0:
                self._line_buffer = ""
            self._consume(chunk)
        return len(text)

    def _consume(self, text: str) -> None:
        if not text:
            return
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            if line:
                self.stream.write(line + "\n")
            else:
                self.stream.write("\n")
            self.stream.flush()

    def flush(self) -> None:
        if self._line_buffer:
            self.stream.write(self._line_buffer)
            self._line_buffer = ""
        self.stream.flush()


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextlib.contextmanager
def with_video_train_log(config):
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        file_stream = LogFileStream(log_handle)
        tee_stdout = TeeStream(sys.stdout, file_stream)
        tee_stderr = TeeStream(sys.stderr, file_stream)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"Logging full CLI output to: {log_path}")
            yield log_path


def prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(f"\n{prompt}")
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option}")
        choice = input("Select option: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid selection.")


def family_dir_name(family: str) -> str:
    return family.replace("+", "_").replace(" ", "_")


def dataset_tag(dataset_scope: str) -> str:
    return {
        "celebdf": "celebdf",
        "ffpp": "ffpp",
        "video_combined": "video_combined",
        "real_ai_videos": "real_ai_videos",
        "video_all": "video_all",
    }[dataset_scope]


def normalize_loss_mode(loss_mode: str | None) -> str:
    mode = (loss_mode or "none").lower()
    aliases = {
        "none": "none",
        "weighted_bce": "weighted_ce",
        "weighted_ce": "weighted_ce",
        "focal": "focal",
        "focal_loss": "focal",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported loss mode: {loss_mode}")
    return aliases[mode]


def loss_tag(loss_mode: str) -> str:
    return {
        "none": "loss-none",
        "weighted_ce": "loss-weighted_ce",
        "focal": "loss-focal",
    }[loss_mode]


def lr_tag(base_lr: float) -> str:
    text = f"{base_lr:.0e}" if base_lr < 1e-3 else f"{base_lr:g}"
    return f"lr-{text.replace('+', '').replace('.', 'p')}"


def build_video_run_config(exp_no: str, dataset_scope: str, batch_size=None, seq_len=None, epochs=None, loss_mode=None, base_lr=None):
    registry = get_video_experiment_registry()
    config = dict(registry[exp_no])
    config["dataset_scope"] = dataset_scope
    config["dataset_names"] = config["dataset_scope_options"][dataset_scope]
    if batch_size is not None:
        config["batch_size"] = batch_size
    if seq_len is not None:
        config["seq_len"] = seq_len
    if epochs is not None:
        config["epochs"] = epochs
    config["loss_mode"] = normalize_loss_mode(loss_mode or config.get("loss_mode", "none"))
    config["seed"] = DEFAULT_SEED
    config["base_lr"] = base_lr if base_lr is not None else 1e-4
    config["weight_decay"] = 1e-4
    config["min_lr"] = 1e-6
    config["warmup_epochs"] = 1
    config["grad_clip"] = 1.0
    config["label_smoothing"] = 0.1 if config["loss_mode"] in {"none", "weighted_ce"} else 0.0
    config["ema_decay"] = 0.999
    config["patience"] = 3
    config["min_delta"] = 1e-4
    config["best_metric"] = "val_f1"
    config["runtime"] = dict(DEFAULT_VIDEO_SMOKE_RUNTIME)
    category_root = CATEGORY_GUIDE[config["category"]]["save_root"]
    run_name = (
        f"{config['experiment_no']}_{config['model_name']}_{dataset_tag(dataset_scope)}_"
        f"{loss_tag(config['loss_mode'])}_{lr_tag(config['base_lr'])}"
    )
    config["run_name"] = run_name
    config["save_dir"] = str(Path(category_root) / family_dir_name(config["family"]) / run_name)
    return config


def run_video_experiment_smoke(config) -> None:
    """Backward-compatible name; current executable path is the real trainer."""

    print("\nCurrent execution path: active video trainer")
    print(
        "Video runtime defaults | "
        f"decode={config['runtime']['decode_backend']} | "
        f"transform={config['runtime']['transform_profile']} | "
        f"gpu_aug={config['runtime']['gpu_aug_mode']} | "
        f"workers={config['runtime']['workers']} | "
        f"prefetch={config['runtime']['prefetch_factor']}"
    )
    train_video_experiment(config)


def configure_runtime() -> None:
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_accuracy(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(y == p) for y, p in zip(labels, preds))
    return correct / len(labels)


def binary_f1(labels: list[int], preds: list[int]) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def apply_warmup(optimizer, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = float(epoch) / float(warmup_epochs)
    warmup_lr = base_lr * scale
    for param_group in optimizer.param_groups:
        param_group["lr"] = warmup_lr


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_history_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
                "lr",
                "seconds",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_split(loader) -> dict:
    samples = loader.dataset.samples
    counts = Counter(label for _, label, _ in samples)
    return {
        "samples": len(samples),
        "batches": len(loader),
        "fake": int(counts.get(0, 0)),
        "real": int(counts.get(1, 0)),
    }


def save_checkpoint(path: Path, model_state: dict, payload: dict) -> None:
    torch.save({"model_state_dict": model_state, **payload}, path)


def format_metric(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


class TensorCacheDataset(Dataset):
    def __init__(self, tensors: list[torch.Tensor], labels: list[int], samples=None) -> None:
        self.tensors = tensors
        self.labels = labels
        self.samples = samples or [("", label, "cached") for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        return loss.mean()


class TemporalConvHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = self.blocks(x).squeeze(-1)
        return self.classifier(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None):
        if state is None:
            shape = (x.shape[0], self.hidden_dim, x.shape[2], x.shape[3])
            h = x.new_zeros(shape)
            c = x.new_zeros(shape)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = (f * c) + (i * g)
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        state = None
        for step in range(feature_maps.shape[1]):
            state = self.cell(feature_maps[:, step], state)
        h, _ = state
        pooled = self.pool(h).flatten(1)
        return self.classifier(pooled)


def compute_class_weights_from_samples(samples) -> torch.Tensor:
    counts = Counter(label for _, label, _ in samples)
    total = sum(counts.values())
    if total == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    weights = []
    for label in (0, 1):
        count = counts.get(label, 0)
        weights.append(total / (2.0 * count) if count > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def preload_tensor_dataset(dataset, desc: str) -> TensorCacheDataset:
    tensors = []
    labels = []
    progress = tqdm(range(len(dataset)), desc=desc, leave=True, dynamic_ncols=True)
    for idx in progress:
        tensor, label = dataset[idx]
        tensors.append(tensor.contiguous())
        labels.append(int(label))
    return TensorCacheDataset(tensors, labels, samples=getattr(dataset, "samples", None))


def maybe_cache_video_loaders(train_loader, val_loader, test_loader, config):
    if config["mode"] == "single":
        print("RAM cache policy: preload train/val/test single-frame tensors before epoch 1")
        cached_train = preload_tensor_dataset(train_loader.dataset, "cache train")
        cached_val = preload_tensor_dataset(val_loader.dataset, "cache val")
        cached_test = preload_tensor_dataset(test_loader.dataset, "cache test")
        return (
            DataLoader(cached_train, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True, drop_last=True),
            DataLoader(cached_val, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True),
            DataLoader(cached_test, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True),
        )

    print("RAM cache policy: preload val/test sequence tensors; keep train sequence streaming")
    cached_val = preload_tensor_dataset(val_loader.dataset, "cache val")
    cached_test = preload_tensor_dataset(test_loader.dataset, "cache test")
    return (
        train_loader,
        DataLoader(cached_val, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(cached_test, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True),
    )


def build_criterion(loss_mode: str, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0):
    if loss_mode == "none":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing), "train_oversample"
    if loss_mode == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=label_smoothing), "loss_weighting"
    return SoftmaxFocalLoss(alpha=class_weights.to(DEVICE), gamma=2.0), "focal_weighting"


class TimmVideoClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, config: dict | None = None) -> None:
        super().__init__()
        config = config or {}
        try:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        except Exception as exc:
            raise RuntimeError(
                f"Video backbone '{model_name}' is not available through the current timm-backed trainer."
            ) from exc
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise RuntimeError(f"Could not determine feature dimension for model '{model_name}'")
        self.temporal_head = str(config.get("temporal_head", "mean")).lower()
        hidden_dim = int(config.get("temporal_hidden_dim", min(feature_dim, 512)))
        num_layers = int(config.get("temporal_layers", 1))
        dropout = float(config.get("temporal_dropout", 0.2))
        if config.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        if feats.shape[1] != getattr(self.backbone, "num_features", feats.shape[1]) and feats.shape[-1] == getattr(self.backbone, "num_features", feats.shape[-1]):
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


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, ema=None, grad_clip=1.0, phase_name="run"):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    preds_all = []
    labels_all = []

    progress = tqdm(loader, leave=True, desc=phase_name, dynamic_ncols=True)
    for inputs, labels in progress:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if inputs.ndim == 4 and DEVICE == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=DEVICE == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        preds_all.extend(int(x) for x in preds.detach().cpu().tolist())
        labels_all.extend(int(x) for x in labels.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": binary_accuracy(labels_all, preds_all),
        "f1": binary_f1(labels_all, preds_all),
    }


def write_run_record(path: Path, config: dict, split_summary: dict, final_summary: dict | None = None) -> None:
    lines = [
        "# Run Record",
        "",
        "## Identity",
        "",
        f"- experiment_no: `{config['experiment_no']}`",
        f"- model_name: `{config['model_name']}`",
        f"- category: `{config['category']}`",
        f"- dataset_scope: `{config['dataset_scope']}`",
        f"- dataset_names: `{config['dataset_names']}`",
        f"- protocol: `{config['protocol']}`",
        f"- loss_mode: `{config['loss_mode']}`",
        f"- temporal_head: `{config.get('temporal_head', 'mean')}`",
        "",
        "## Training Config",
        "",
        f"- epochs: `{config['epochs']}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- seq_len: `{config['seq_len']}`",
        f"- base_lr: `{config['base_lr']}`",
        f"- min_lr: `{config['min_lr']}`",
        f"- weight_decay: `{config['weight_decay']}`",
        f"- ema_decay: `{config['ema_decay']}`",
        f"- patience: `{config['patience']}`",
        f"- decode_backend: `{config['runtime']['decode_backend']}`",
        f"- workers: `{config['runtime']['workers']}`",
        f"- prefetch_factor: `{config['runtime']['prefetch_factor']}`",
        "",
        "## Split Summary",
        "",
    ]
    for split_name in ("train", "val", "test"):
        split = split_summary[split_name]
        lines.extend(
            [
                f"### {split_name.title()}",
                "",
                f"- samples: `{split['samples']}`",
                f"- batches: `{split['batches']}`",
                f"- fake: `{split['fake']}`",
                f"- real: `{split['real']}`",
                "",
            ]
        )
    if final_summary is not None:
        lines.extend(
            [
                "## Final Result",
                "",
                f"- best_epoch: `{final_summary['best_epoch']}`",
                f"- best_val_f1: `{format_metric(final_summary['best_val_f1'])}`",
                f"- best_val_acc: `{format_metric(final_summary['best_val_acc'])}`",
                f"- test_loss: `{final_summary['test_metrics']['loss']:.6f}`",
                f"- test_acc: `{final_summary['test_metrics']['acc']:.6f}`",
                f"- test_f1: `{final_summary['test_metrics']['f1']:.6f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_video_experiment(config: dict) -> dict:
    set_seed(config["seed"])
    configure_runtime()

    runtime = config["runtime"]
    os.environ["DF_VIDEO_DECODE_BACKEND"] = runtime["decode_backend"]
    os.environ["DF_NUM_WORKERS"] = str(runtime["workers"])
    os.environ["DF_PREFETCH_FACTOR"] = str(runtime["prefetch_factor"])
    os.environ["DF_VIDEO_READER_CACHE"] = str(runtime["reader_cache"])
    os.environ.pop("DF_FFMPEG_OUTPUT_SIZE", None)

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "config.json"
    history_path = save_dir / "history.csv"
    best_summary_path = save_dir / "best_summary.json"
    final_summary_path = save_dir / "final_summary.json"
    split_summary_path = save_dir / "split_summary.json"
    run_record_path = save_dir / "run_record.md"
    best_path = save_dir / "best.pth"
    last_path = save_dir / "last.pth"

    builder = DatasetBuilder(root="datasets", seed=config["seed"])
    train_loader, val_loader, test_loader = builder.get_loaders(
        batch_size=config["batch_size"],
        mode=config["mode"],
        seq_len=config["seq_len"],
        dtype="video",
        protocol="video_only",
        dataset_names=config["dataset_names"],
        clip_sampling_train="random",
        clip_sampling_eval="center",
        balanced=(config["loss_mode"] == "none"),
    )

    split_summary = {
        "train": summarize_split(train_loader),
        "val": summarize_split(val_loader),
        "test": summarize_split(test_loader),
    }
    train_loader, val_loader, test_loader = maybe_cache_video_loaders(train_loader, val_loader, test_loader, config)
    save_json(config_path, config)
    save_json(split_summary_path, split_summary)
    write_run_record(run_record_path, config, split_summary)

    print(f"Device: {DEVICE}")
    print(f"Run: {config['run_name']}")
    print(f"Save dir: {save_dir}")
    print(f"Datasets: {config['dataset_names']}")

    model = TimmVideoClassifier(config["model_name"], config=config).to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    class_weights = compute_class_weights_from_samples(train_loader.dataset.samples)
    criterion, imbalance_strategy = build_criterion(config["loss_mode"], class_weights, label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["base_lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["min_lr"])
    scaler = GradScaler("cuda", enabled=DEVICE == "cuda")
    ema = EMA(model, config["ema_decay"])

    print(f"Loss mode: {config['loss_mode']}")
    print(f"Imbalance strategy: {imbalance_strategy}")
    print(f"Train class weights: fake={class_weights[0]:.4f} | real={class_weights[1]:.4f}")

    best_f1 = float("-inf")
    best_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n===== EPOCH {epoch}/{config['epochs']} =====")
        epoch_start = time.time()
        apply_warmup(optimizer, epoch, config["warmup_epochs"], config["base_lr"])

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            grad_clip=config["grad_clip"],
            phase_name=f"train e{epoch}",
        )

        eval_model = ema.ema if ema is not None else model
        val_metrics = run_epoch(eval_model, val_loader, criterion, phase_name=f"val e{epoch}")
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch > config["warmup_epochs"]:
            scheduler.step()
        elapsed = time.time() - epoch_start

        print(
            f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['acc']:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"lr={current_lr:.7f} | time={elapsed:.1f}s"
        )

        append_history_row(
            history_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_acc": f"{train_metrics['acc']:.6f}",
                "train_f1": f"{train_metrics['f1']:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_acc": f"{val_metrics['acc']:.6f}",
                "val_f1": f"{val_metrics['f1']:.6f}",
                "lr": f"{current_lr:.8f}",
                "seconds": f"{elapsed:.4f}",
            },
        )

        improved = val_metrics["f1"] > (best_f1 + config["min_delta"])
        if improved:
            best_f1 = val_metrics["f1"]
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state = deepcopy(eval_model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                best_state,
                {
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_f1": val_metrics["f1"],
                    "config": config,
                },
            )
            save_json(
                best_summary_path,
                {
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_f1": val_metrics["f1"],
                },
            )
            print("best.pth updated")
        else:
            epochs_without_improvement += 1
            print(f"no improvement | patience={epochs_without_improvement}/{config['patience']}")
            if epochs_without_improvement >= config["patience"]:
                print("early stopping triggered")
                break

    save_checkpoint(
        last_path,
        (ema.ema if ema is not None else model).state_dict(),
        {"epoch": epoch, "config": config},
    )

    test_model = deepcopy(model)
    if best_state is not None:
        test_model.load_state_dict(best_state)
    test_model = test_model.to(DEVICE)
    test_metrics = run_epoch(test_model, test_loader, criterion, phase_name="test")
    print(f"\nTEST | loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.4f} | f1={test_metrics['f1']:.4f}")

    final_summary = {
        "experiment_no": config["experiment_no"],
        "model_name": config["model_name"],
        "category": config["category"],
        "dataset_scope": config["dataset_scope"],
        "dataset_names": config["dataset_names"],
        "loss_mode": config["loss_mode"],
        "best_epoch": best_epoch if best_epoch else None,
        "best_val_f1": best_f1 if best_f1 != float("-inf") else None,
        "best_val_acc": best_acc if best_epoch else None,
        "test_metrics": test_metrics,
        "save_dir": str(save_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    save_json(final_summary_path, final_summary)
    write_run_record(run_record_path, config, split_summary, final_summary=final_summary)
    return final_summary
