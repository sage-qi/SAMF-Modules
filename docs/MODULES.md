# Released module interfaces and defaults

All values below describe the selected supplied implementations. They are not a
claim about dataset-specific settings in the manuscript. Use the JSON files in
`examples/` for fully specified synthetic demonstration settings.

## Encoder interface

Each released encoder accepts `input_dim`, the flattened width of a sample. The
assembly wrapper supplies it as `T*C`. Encoders return `[B, output_dim]`.

| Encoder | Constructor parameters other than `input_dim` | Output width |
| --- | --- | --- |
| `MLPEncoder` | `output_dim=32`, `hidden_dim=60`, `layers_num=3`, `dropout=0.2`, `activation="tanh"` | `output_dim` |
| `CNNEncoder` | `seq_len=None`, `out_channels=64`, `kernel_size=3`, `dropout=0.1`, `projection_dim=64` | `projection_dim` |
| `LSTMEncoder` | `seq_len=None`, `hidden_dim=128`, `layers=1`, `dropout=0.0`, `projection_dim=64` | `projection_dim` |
| `SCCNNLSTMEncoder` | `seq_len=None`, `cnn_out_channels=32`, `cnn_kernel_size=3`, `cnn_dropout=0.1`, `lstm_hidden_dim=64`, `lstm_layers=1`, `lstm_dropout=0.0`, `projection_dim=64` | `projection_dim` |

`MLPEncoder.layers_num` counts linear layers and must be at least two. The retained
activation implementation selects ReLU for `"relu"` and otherwise uses Tanh; use
`"relu"` or `"tanh"` explicitly instead of relying on the source fallback.

Sequential encoders interpret features as `[B, seq_len, input_dim/seq_len]`. The
standalone builder fills `seq_len=T` when it is omitted. Direct construction of a
sequential encoder with `seq_len=None` instead falls back to `seq_len=input_dim`,
as in the supplied implementation. Specify `seq_len` explicitly outside the builder.

The CNN branch applies one Conv1d, ReLU, Dropout, and adaptive average pooling. The
LSTM encoder uses its final hidden output and a Linear/ReLU projection. In the
fusion encoder, both branches receive the same full feature sequence; their
vectors are concatenated and projected. The class name is retained from the source.
LSTM dropout is applied only when the number of recurrent layers exceeds one.

## Predictor interface

The wrapper provides predictor `input_dim=A`, the aligned embedding width. The
output is always `[B, 1]`. These scalar heads do not produce future trajectories.

| Predictor | Constructor parameters other than `input_dim` | Computation |
| --- | --- | --- |
| `MLPPredictor` | `hidden_dim=32`, `dropout=0.2` | Dropout, Linear, ReLU, Linear |
| `TransformerPredictor` | `seq_len=1`, `d_model=128`, `nhead=4`, `num_layers=2`, `dim_feedforward=256`, `dropout=0.1` | Embedding grouping, input projection, TransformerEncoder, mean pooling, normalized regression head |
| `AttentivePoolingPredictor` | required `seq_len`, `attn_hidden_dim=128`, `projection_dim=128`, `dropout=0.1` | Learned attention weights over embedding groups, weighted pooling, regression head |

For grouped heads, `A % seq_len` must be zero. The group width is `A/seq_len`.
Transformer `d_model` must also be divisible by `nhead`. The attention/Transformer
heads reshape an encoder vector into groups; they do not restore cycle-wise
representations lost through encoder pooling. The Transformer implementation is
an encoder-based scalar head, not a TransformerDecoder or autoregressive decoder.

## Assembly configuration

The root configuration contains:

| Key | Meaning |
| --- | --- |
| `description` | Optional human-readable description; ignored by the builder |
| `input.sequence_length` | Positive integer `T` for the synthetic feature sequence |
| `input.feature_channels` | Positive integer `C` for features at each sequence position |
| `alignment_dim` | Positive integer `A`; defaults to 64 |
| `encoder.name` / `encoder.params` | Explicit encoder name and constructor parameters |
| `predictor.name` / `predictor.params` | Explicit predictor name and constructor parameters |

The builder derives each `input_dim`. It connects the encoder to the predictor
through `nn.Identity` when `E=A`, or `nn.Linear(E, A)` otherwise. The default MLP
example exercises the Linear branch; the other examples use matching widths.

## Registering an additional module

```python
from torch import nn
from samf_modules import ENCODERS

@ENCODERS.register()
class ExampleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, feature):
        return self.projection(feature)
```

Register the class in the same Python process before calling `build_model`, then
select `"ExampleEncoder"` in the configuration. A custom encoder must accept the
derived `input_dim` and return a vector with its declared `output_dim`. Additional
constructor parameters belong in `encoder.params`. New predictor classes can be
registered through `PREDICTORS`; they must accept `input_dim` and return `[B, 1]`.

Registration only makes a module available for explicit construction. It does not
add a PG-EPM rule or automatically establish the module's suitability for any dataset.
