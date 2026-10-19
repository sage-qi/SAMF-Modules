# Source provenance and extraction changes

## Supplied source snapshot

The source folder was identified as `microsoft-BatteryML-235369c`. This identifier
names the supplied local snapshot; this release does not assert that its locally
modified files are identical to an upstream Git revision.

Only the following module-library files were used as sources for the released
architectures:

- `batteryml/models/rul_predictors/encoder_lib/__init__.py`
- `batteryml/models/rul_predictors/predictor_lib/__init__.py`

The supplied root `LICENSE` was retained. No source paths tied to an individual
computer are required by the standalone package.

## Extracted modules

| Released file | Retained source definitions |
| --- | --- |
| `samf_modules/encoders.py` | `_reshape_sequence`, `BaseEncoder`, `MLPEncoder`, `_ConvBlock`, `_CNNLSTMEncoder`, `SCCNNLSTMEncoder`, `CNNEncoder`, `LSTMEncoder` |
| `samf_modules/predictors.py` | `BasePredictor`, `MLPPredictor`, `TransformerPredictor`, `AttentivePoolingPredictor` |

Other modules from those source files are deliberately omitted. This is not the
full module collection described in the manuscript.

## Adaptations

1. Registry imports now point to `samf_modules.registry`. There is no import of
   BatteryML's model base class, training wrapper, pipeline, or dataset objects.
2. Tensor `.view(...)` calls were changed to `.reshape(...)` so the excerpts can
   also handle noncontiguous input tensors without requiring a view.
3. The source Transformer's automatic CUDA availability check, training-specific
   scaled-dot-product-attention backend override, and double-backward marker were
   removed. Its projection, TransformerEncoder layers, pooling, and regression
   architecture are retained. The demonstration runs on CPU and makes no claim
   about reproducing training-time kernels or higher-order derivatives.
4. Descriptions of the MLP encoder and `SCCNNLSTMEncoder` were adjusted to describe
   the retained implementation without assuming the unavailable training context
   or a capacity-only branch.
5. A standalone registry, explicit configuration builder, linear dimension-alignment
   wrapper, synthetic demo, and illustrative JSON files were added. They are
   release adapters, not a reimplementation of PG-EPM or the full SAMF pipeline.
6. The extracted annotations use `Optional[int]` instead of `int | None` for the
   declared Python 3.9 baseline. Constructor behavior is otherwise retained.

## Excluded material

No source files from data processing, feature extraction, feature profiling,
configuration matching/search, constrained losses, training, testing, notebooks,
workspaces, experiment configurations, results, caches, or GUI directories were
copied into this release. No pretrained weights or real battery records are present.

The supplied subset did not provide a complete correspondence between these
implementation class names and all 120 manuscript configuration indices. No such
correspondence, trained performance, or exhaustive reproducibility is claimed here.

## Preparation status

Preparation consisted of reading source, extracting selected definitions, writing
the standalone adapters and documentation, and static file inspection. The package
was not imported or executed. No installation, demo run, compile command, training,
benchmark, or automated test was performed.
