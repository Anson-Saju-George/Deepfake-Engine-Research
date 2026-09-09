"""Shared temporal head modules (FIXTURE_PLAN.md T2b).

Extracted from train/video/video_train_backbone.py -- these classes were
duplicated byte-for-byte in profile_model_complexity.py (confirmed identical
before this extraction), which imported nothing from the trainer to avoid
the (now-fixed) contextlib/sys import bug that used to make importing the
trainer module crash. Both now import from here instead.

Bit-identical behavior after extraction is a hard requirement, not a nice-to-have:
profile_model_complexity.py's already-published docs/MODEL_COMPLEXITY.md numbers
(param counts, FLOPs) must not silently drift just because these classes moved.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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
