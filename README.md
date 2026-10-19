# SAMF Selected Module Library

This repository releases selected encoder and predictor implementations related to
**SAMF: A Structure-Adaptive Modular Framework for Remaining Useful Life Prediction
of Lithium-Ion Batteries**. It demonstrates module registration, explicit model
assembly, dimensional alignment, and forward computation using synthetic inputs.

This is a **partial module-library release**, not the complete SAMF framework or a
reproduction package for the paper's experiments. The full framework and the
training and testing code are not included because of laboratory intellectual
property restrictions. No battery records, experiment outputs, fitted weights,
or private configuration files are distributed.

[中文说明](README.zh-CN.md)

## Released components

| Component | Contents |
| --- | --- |
| Encoder subset | `MLPEncoder`, `CNNEncoder`, `LSTMEncoder`, `SCCNNLSTMEncoder` |
| Predictor subset | `MLPPredictor`, `TransformerPredictor`, `AttentivePoolingPredictor` |
| Assembly | Local registries and an encoder/projection/predictor wrapper |
| Demonstration | CPU forward computation on generated random input |
| Examples | Four JSON configurations that explicitly choose modules |
| Documentation | Tensor interfaces, constructor defaults, adaptations, and source provenance |

The four released encoders and three released predictors expose a smaller
composition space than the 12 encoders, 10 predictors, and 120 configurations
described in the manuscript. The example configurations are illustrative settings,
not the dataset-specific configurations or numbered configurations in the paper.

Not released: feature extraction, seven-dimensional profiling, the PG-EPM matching
function, configuration search, physical/monotonic loss functions, fitting loops,
dataset processing or partitioning, benchmark evaluation, or the graphical user
interface. The demo performs **no automatic configuration recommendation**.

## Installation

Use Python 3.9 or later and a compatible PyTorch 2.x installation. The demonstration
uses the CPU and does not require CUDA or the original BatteryML package. Standard
library modules handle the JSON configurations.

From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the runtime dependency:

```bash
python -m pip install -r requirements.txt
```

Choose a PyTorch build supported by your Python version and operating system. The
dependency range is an installation constraint, not a claim that every version in
that range has been validated. Editable installation is optional if importing the
package from outside the repository:

```bash
python -m pip install -e .
```

## Demonstration commands

```bash
python demo.py --list-modules
python demo.py --config examples/mlp_mlp.json
python demo.py --config examples/cnn_transformer.json
python demo.py --config examples/lstm_attention.json
python demo.py --config examples/fusion_mlp.json
```

The demo uses random initialization and random feature tensors. It does not load
data or weights and does not fit parameters, compute prediction metrics, or
reproduce RMSE/MAE/MAPE or timing results from the paper. Calling `model.eval()`
only disables training-mode behavior such as dropout; it is not a benchmark or
test-set evaluation.

Each supplied configuration uses input shape `[4, 16, 8]` by default and is designed
to produce scalar outputs with shape `[4, 1]`. The program prints the actual output
shape, selected modules, dimensions, alignment layer, and parameter count. The MLP
example uses a 32-dimensional encoder output followed by a linear projection to
64 dimensions. The other examples use an identity alignment when dimensions match.
These are expected structural properties, not captured execution results.

Batch size and the random seed can be changed explicitly:

```bash
python demo.py --config examples/mlp_mlp.json --batch-size 2 --seed 7
```

A seed makes the input and initialization explicit; it does not establish
cross-version numerical equivalence or reproduce trained predictions.

## Tensor interface and parameters

```text
Synthetic features [B, T, C]
    -> flatten [B, T*C]
    -> selected encoder [B, E]
    -> linear projection or identity [B, A]
    -> selected predictor [B, 1]
```

Here `T` is `input.sequence_length`, `C` is `input.feature_channels`, `E` is the
encoder's `output_dim`, and `A` is `alignment_dim`. A flattened tensor `[B, T*C]`
is also accepted. These dimensions refer to a demonstration feature tensor; no
raw-cycle feature extraction is implemented.

The builder derives the encoder's `input_dim = T*C` and the predictor's
`input_dim = A`; do not specify either inside `params`. For the released sequential
encoders, `seq_len` must equal `T`. For the Transformer and attentive-pooling heads,
`seq_len` is the number of groups formed by reshaping the aligned embedding, so it
must divide `A`. It is **not** automatically the number of original battery cycles.

The source `TransformerPredictor` uses `torch.nn.TransformerEncoder` layers,
mean pooling, and a scalar regression head. It is not an autoregressive Transformer
decoder or a sequence-to-sequence output implementation. Similarly, the released
`SCCNNLSTMEncoder` passes the full feature sequence through parallel CNN and LSTM
branches. Its source-library name is retained without implying a separate
capacity-only input channel.

See [module details](docs/MODULES.md) for all released constructor defaults and
[source provenance](docs/PROVENANCE.md) for changes made during extraction.

## Python usage

```python
import json
import torch
from samf_modules import build_model

with open("examples/mlp_mlp.json", encoding="utf-8") as handle:
    config = json.load(handle)

model = build_model(config)
model.eval()
synthetic_features = torch.randn(4, 16, 8)
with torch.no_grad():
    scalar_outputs = model(synthetic_features)
```

Module choices in this example come from the JSON file, not from PG-EPM. Scalar
outputs from untrained parameters have no calibrated RUL units or predictive value.

## Repository layout

```text
.
|-- README.md
|-- README.zh-CN.md
|-- LICENSE
|-- NOTICE.md
|-- requirements.txt
|-- pyproject.toml
|-- demo.py
|-- examples/
|   |-- mlp_mlp.json
|   |-- cnn_transformer.json
|   |-- lstm_attention.json
|   `-- fusion_mlp.json
|-- samf_modules/
|   |-- __init__.py
|   |-- registry.py
|   |-- encoders.py
|   |-- predictors.py
|   `-- model.py
`-- docs/
    |-- MODULES.md
    `-- PROVENANCE.md
```

## License and execution status

The supplied module files carry Microsoft Corporation copyright notices and the
MIT License. These notices are retained; see [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md). This release does not assert that Microsoft authored or
endorses SAMF.

This package was prepared by static source extraction and review. Its demo,
imports, installation, and forward computations were **not executed or tested**
during preparation. No runtime verification or experimental reproduction is claimed.
