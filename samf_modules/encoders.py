# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
# Extracted and adapted from the supplied BatteryML-derived module library.
# See docs/PROVENANCE.md for the release scope and adaptations.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .registry import ENCODERS


def _reshape_sequence(feature: torch.Tensor,
                      seq_len: Optional[int]) -> torch.Tensor:
    """Ensure feature has shape [B, seq_len, D]."""
    if feature.ndim == 3:
        if seq_len is not None and feature.size(1) != seq_len:
            raise ValueError('Feature sequence length mismatch.')
        return feature
    if seq_len is None:
        raise ValueError('seq_len must be provided when feature is 2D.')
    if feature.ndim > 3:
        feature = feature.reshape(feature.size(0), -1)
    if feature.ndim == 1:
        feature = feature.unsqueeze(0)
    B, F = feature.shape
    if F % seq_len != 0:
        raise ValueError('Input dimension cannot be evenly split by seq_len.')
    step_dim = F // seq_len
    return feature.reshape(B, seq_len, step_dim)


class BaseEncoder(nn.Module):
    """Common encoder interface."""

    output_dim: int

    def __init__(self):
        super().__init__()
        self.output_dim = 0


@ENCODERS.register()
class MLPEncoder(BaseEncoder):
    """MLP encoder that returns one vector per sample."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 32,
                 hidden_dim: int = 60,
                 layers_num: int = 3,
                 dropout: float = 0.2,
                 activation: str = 'tanh'):
        super().__init__()
        if layers_num < 2:
            raise ValueError('layers_num must be >= 2')
        act = getattr(torch, activation.lower(), torch.tanh)

        modules = []
        in_channels = input_dim
        for idx in range(layers_num - 1):
            modules.append(nn.Linear(in_channels, hidden_dim))
            modules.append(nn.Dropout(dropout))
            modules.append(nn.ReLU() if act == torch.relu else nn.Tanh())
            in_channels = hidden_dim
        modules.append(nn.Linear(in_channels, output_dim))

        self.net = nn.Sequential(*modules)
        self.output_dim = output_dim

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim > 2:
            feature = feature.reshape(feature.size(0), -1)
        return self.net(feature.float())


class _ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _CNNLSTMEncoder(BaseEncoder):
    """Base encoder that feeds the entire feature sequence to both CNN and LSTM."""

    def __init__(self,
                 input_dim: int,
                 seq_len: Optional[int] = None,
                 cnn_out_channels: int = 32,
                 cnn_kernel_size: int = 3,
                 cnn_dropout: float = 0.1,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0,
                 projection_dim: int = 64):
        super().__init__()
        if seq_len is None:
            seq_len = input_dim
        if seq_len <= 0:
            raise ValueError('seq_len must be positive.')
        if input_dim % seq_len != 0:
            raise ValueError('input_dim must be divisible by seq_len.')

        self.seq_len = seq_len
        self.per_step_dim = max(1, input_dim // seq_len)

        self.cnn = _ConvBlock(
            in_channels=self.per_step_dim,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout
        )

        self.lstm = nn.LSTM(
            input_size=self.per_step_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels + lstm_hidden_dim, projection_dim),
            nn.ReLU()
        )
        self.output_dim = projection_dim

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        seq = _reshape_sequence(feature.float(), self.seq_len)
        cnn_input = seq.permute(0, 2, 1)  # [B, per_step_dim, seq_len]
        cnn_feat = self.cnn(cnn_input)
        lstm_output, _ = self.lstm(seq)
        lstm_feat = lstm_output[:, -1, :]
        return self.fc(torch.cat([cnn_feat, lstm_feat], dim=1))


@ENCODERS.register()
class SCCNNLSTMEncoder(_CNNLSTMEncoder):
    """CNN/LSTM fusion encoder retained under its source-library name."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)


@ENCODERS.register()
class CNNEncoder(BaseEncoder):
    """CNN-only encoder over the full feature sequence."""

    def __init__(self,
                 input_dim: int,
                 seq_len: Optional[int] = None,
                 out_channels: int = 64,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 projection_dim: int = 64):
        super().__init__()
        if seq_len is None:
            seq_len = input_dim
        if seq_len <= 0:
            raise ValueError('seq_len must be positive.')
        if input_dim % seq_len != 0:
            raise ValueError('input_dim must be divisible by seq_len.')

        self.seq_len = seq_len
        self.per_step_dim = max(1, input_dim // seq_len)
        self.cnn = _ConvBlock(
            in_channels=self.per_step_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, projection_dim),
            nn.ReLU()
        )
        self.output_dim = projection_dim

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        seq = _reshape_sequence(feature.float(), self.seq_len)
        cnn_input = seq.permute(0, 2, 1)
        cnn_feat = self.cnn(cnn_input)
        return self.fc(cnn_feat)


@ENCODERS.register()
class LSTMEncoder(BaseEncoder):
    """LSTM-only encoder over the full feature sequence."""

    def __init__(self,
                 input_dim: int,
                 seq_len: Optional[int] = None,
                 hidden_dim: int = 128,
                 layers: int = 1,
                 dropout: float = 0.0,
                 projection_dim: int = 64):
        super().__init__()
        if seq_len is None:
            seq_len = input_dim
        if seq_len <= 0:
            raise ValueError('seq_len must be positive.')
        if input_dim % seq_len != 0:
            raise ValueError('input_dim must be divisible by seq_len.')

        self.seq_len = seq_len
        self.per_step_dim = max(1, input_dim // seq_len)
        self.lstm = nn.LSTM(
            input_size=self.per_step_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU()
        )
        self.output_dim = projection_dim

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        seq = _reshape_sequence(feature.float(), self.seq_len)
        output, _ = self.lstm(seq)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)


__all__ = ['BaseEncoder', 'MLPEncoder', 'SCCNNLSTMEncoder', 'CNNEncoder', 'LSTMEncoder']
