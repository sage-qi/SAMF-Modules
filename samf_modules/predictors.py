# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
# Extracted and adapted from the supplied BatteryML-derived module library.
# See docs/PROVENANCE.md for the release scope and adaptations.

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import PREDICTORS


class BasePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim


@PREDICTORS.register()
class MLPPredictor(BasePredictor):
    """MLP regressor head."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim > 2:
            embedding = embedding.reshape(embedding.size(0), -1)
        return self.net(embedding).reshape(-1, 1)


@PREDICTORS.register()
class TransformerPredictor(BasePredictor):
    """Predictor that feeds embeddings through a Transformer encoder."""

    def __init__(self,
                 input_dim: int,
                 seq_len: int = 1,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__(input_dim)
        if seq_len <= 0:
            raise ValueError('seq_len must be positive.')
        if input_dim % seq_len != 0:
            raise ValueError('input_dim must be divisible by seq_len.')
        self.seq_len = seq_len
        self.token_dim = input_dim // seq_len

        self.input_proj = nn.Linear(self.token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 2:
            embedding = embedding.reshape(embedding.size(0),
                                       self.seq_len,
                                       self.token_dim)
        elif embedding.ndim == 3:
            if embedding.size(1) != self.seq_len:
                raise ValueError('Embedding seq_len mismatch.')
            if embedding.size(2) != self.token_dim:
                raise ValueError('Embedding token_dim mismatch.')
        else:
            embedding = embedding.reshape(embedding.size(0), self.seq_len,
                                       self.token_dim)
        emb = self.input_proj(embedding.float())
        encoded = self.encoder(emb).mean(dim=1)
        return self.regressor(encoded).reshape(-1, 1)


@PREDICTORS.register()
class AttentivePoolingPredictor(BasePredictor):
    """Predictor with learnable attention pooling over the sequence."""

    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 attn_hidden_dim: int = 128,
                 projection_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(input_dim)
        if seq_len <= 0:
            raise ValueError('seq_len must be positive.')
        if input_dim % seq_len != 0:
            raise ValueError('input_dim must be divisible by seq_len.')
        self.seq_len = seq_len
        self.token_dim = input_dim // seq_len

        self.score_net = nn.Sequential(
            nn.Linear(self.token_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.token_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 1)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 2:
            embedding = embedding.reshape(embedding.size(0), self.seq_len, self.token_dim)
        seq = embedding.float()
        scores = self.score_net(seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(self.dropout(seq) * weights, dim=1)
        return self.regressor(pooled).reshape(-1, 1)


__all__ = ['BasePredictor', 'MLPPredictor', 'TransformerPredictor', 'AttentivePoolingPredictor']
