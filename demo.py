"""Demonstrate module assembly on synthetic inputs, without fitting or testing."""

import argparse
import json
from pathlib import Path

import torch

from samf_modules import ENCODERS, PREDICTORS, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "examples" / "mlp_mlp.json",
        help="A JSON file that explicitly selects the modules and their parameters.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-modules", action="store_true")
    args = parser.parse_args()

    if args.list_modules:
        print(json.dumps({"encoders": ENCODERS.names, "predictors": PREDICTORS.names}, indent=2))
        return
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    torch.manual_seed(args.seed)
    model = build_model(config).cpu()
    model.eval()  # Disables dropout; does not load fitted weights or evaluate accuracy.
    features = torch.randn(
        args.batch_size, model.sequence_length, model.feature_channels
    )
    with torch.no_grad():
        output = model(features)

    summary = {
        "purpose": "Synthetic forward-pass demonstration with untrained parameters",
        "encoder": config["encoder"]["name"],
        "predictor": config["predictor"]["name"],
        "device": "cpu",
        "seed": args.seed,
        "input_shape": list(features.shape),
        "flattened_input_dim": model.input_dim,
        "encoder_output_dim": model.encoder_output_dim,
        "alignment_dim": model.alignment_dim,
        "alignment_layer": type(model.alignment).__name__,
        "output_shape": list(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_interpretation": "Uncalibrated scalar outputs; no RUL or accuracy claim",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
