"""Explicit encoder/projection/predictor assembly for forward demonstrations."""

from typing import Any, Mapping

import torch
from torch import nn

from .registry import ENCODERS, PREDICTORS


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _module_specification(config: Mapping[str, Any], key: str):
    specification = config.get(key)
    if not isinstance(specification, Mapping):
        raise ValueError(f"{key} must contain a module name and optional params.")
    name = specification.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{key}.name must be a nonempty string.")
    parameters = specification.get("params", {})
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{key}.params must be a JSON object.")
    if "input_dim" in parameters:
        raise ValueError(f"{key}.params.input_dim is derived by the model builder.")
    return name, dict(parameters)


class ModularRegressor(nn.Module):
    """Connect selected modules; no feature extraction, recommendation, or fitting.

    Input: [batch, sequence_length, feature_channels], or its flattened form.
    Output: [batch, 1]. Outputs have no calibrated RUL meaning without training.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        sequence_length: int,
        feature_channels: int,
        alignment_dim: int,
    ):
        super().__init__()
        self.sequence_length = _positive_integer(sequence_length, "sequence_length")
        self.feature_channels = _positive_integer(feature_channels, "feature_channels")
        self.input_dim = self.sequence_length * self.feature_channels
        self.alignment_dim = _positive_integer(alignment_dim, "alignment_dim")
        self.encoder = encoder
        self.predictor = predictor
        self.encoder_output_dim = _positive_integer(encoder.output_dim, "encoder.output_dim")
        if self.encoder_output_dim == self.alignment_dim:
            self.alignment = nn.Identity()
        else:
            self.alignment = nn.Linear(self.encoder_output_dim, self.alignment_dim)

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            expected = (self.sequence_length, self.feature_channels)
            if tuple(features.shape[1:]) != expected:
                raise ValueError(f"Expected sequence/channel dimensions {expected}.")
        elif features.ndim == 2:
            if features.shape[1] != self.input_dim:
                raise ValueError(f"Expected flattened input width {self.input_dim}.")
        else:
            raise ValueError("Expected a 2D flattened batch or a 3D feature batch.")
        if features.shape[0] <= 0:
            raise ValueError("The batch must contain at least one sample.")
        return features.float().reshape(features.shape[0], self.input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened = self._flatten_features(features)
        embedding = self.encoder(flattened)
        aligned = self.alignment(embedding)
        return self.predictor(aligned)


def build_model(config: Mapping[str, Any]) -> ModularRegressor:
    """Build a user-specified configuration, without searching or recommending it."""
    if not isinstance(config, Mapping):
        raise ValueError("The model configuration must be a JSON object.")
    feature_specification = config.get("input")
    if not isinstance(feature_specification, Mapping):
        raise ValueError("The configuration must contain an input object.")
    sequence_length = _positive_integer(
        feature_specification.get("sequence_length"), "input.sequence_length"
    )
    feature_channels = _positive_integer(
        feature_specification.get("feature_channels"), "input.feature_channels"
    )
    input_dim = sequence_length * feature_channels
    alignment_dim = _positive_integer(config.get("alignment_dim", 64), "alignment_dim")

    encoder_name, encoder_parameters = _module_specification(config, "encoder")
    if encoder_name in {"CNNEncoder", "LSTMEncoder", "SCCNNLSTMEncoder"}:
        encoder_parameters.setdefault("seq_len", sequence_length)
        if encoder_parameters["seq_len"] != sequence_length:
            raise ValueError("encoder.params.seq_len must match input.sequence_length.")
    encoder_parameters["input_dim"] = input_dim
    encoder = ENCODERS.build(encoder_name, encoder_parameters)

    predictor_name, predictor_parameters = _module_specification(config, "predictor")
    predictor_parameters["input_dim"] = alignment_dim
    predictor = PREDICTORS.build(predictor_name, predictor_parameters)
    return ModularRegressor(
        encoder, predictor, sequence_length, feature_channels, alignment_dim
    )
