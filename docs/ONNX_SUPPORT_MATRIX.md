# ONNX Export Support Matrix

Canonical list of which AiDotNet layer types are supported by the new protobuf-based ONNX export path (`AiModelResult.ExportToOnnx`). Updated whenever a new converter lands.

| Status | Meaning |
|---|---|
| ✅ | Supported — round-trip tested against `Microsoft.ML.OnnxRuntime.InferenceSession`, output matches AiDotNet within 1e-5 |
| 🚧 | Planned for an upcoming PR |
| ❌ | Not yet planned. Calling `ConvertToOnnx` throws `OnnxExportUnsupportedException` with the layer's type name |

## v0.1 (current — `feature/onnx-export`)

| Layer | ONNX op(s) emitted | Notes |
|---|---|---|
| `DenseLayer<T>` | `Gemm` (+ optional activation op) | If the layer was constructed with an embedded activation, the activation is emitted as a chained op after the Gemm. Supported embedded activations: Identity, ReLU, Sigmoid, Tanh, Softmax. Float32 element type only in v0.1. |
| `ActivationLayer<T>` | `Relu` / `Sigmoid` / `Tanh` / `Softmax` / (no-op for Identity) | Standalone activation layer between other layers. |
| `BatchNormalizationLayer<T>` | `BatchNormalization` | Uses running stats (inference mode). Emits 4 initializers: scale (γ), B (β), mean (running_mean), var (running_var). The configured `epsilon` becomes the op's `epsilon` attribute. |
| `DropoutLayer<T>` | `Identity` | Dropout is a no-op outside training; Identity preserves graph topology without affecting output. |
| ✅ Status overall | Sequential models composed of the above layers, float32. | Demo target shape supported: Dense → Activation → Dense → Activation. |

## v0.2 (next — not yet started)

| Layer | Status | Likely ONNX op(s) |
|---|---|---|
| `ConvolutionalLayer<T>` (2D Conv) | 🚧 | `Conv` |
| `Conv3DLayer<T>` | 🚧 | `Conv` |
| `MaxPoolingLayer<T>` | 🚧 | `MaxPool` |
| `AveragePoolingLayer<T>` | 🚧 | `AveragePool` |
| `AdaptiveAveragePoolingLayer<T>` | 🚧 | `GlobalAveragePool` (when adaptive output size = 1) |
| `FlattenLayer<T>` | 🚧 | `Flatten` |
| `EmbeddingLayer<T>` | 🚧 | `Gather` |

## v0.3+ (deferred — open scope)

| Family | Examples | Why deferred |
|---|---|---|
| Recurrent | `LSTMLayer`, `GRULayer`, `RecurrentLayer` | ONNX RNN/LSTM/GRU ops have non-trivial parameter packing conventions; needs its own design pass. |
| Attention / Transformer | `MultiHeadAttentionLayer`, `TransformerBlock`, `RotaryPositionalEncodingLayer` | Spec-sensitive; opset version matters more here. ONNX has multiple representations (`Attention` op, decomposed `MatMul`/`Softmax`/`Mul` chain). |
| Vision-Language / Multimodal | `JanusPro`, `LayoutXLM`, `Donut`, `Helix` | Big composite models; export at model-level rather than layer-level. |
| Quantized / sparse | `QuantizedLinearLayer`, sparse variants | Needs `QuantizeLinear` / `DequantizeLinear` op support and quantization metadata. |
| Custom / user-defined | Anything outside `AiDotNet.NeuralNetworks.Layers` | Out of scope. Users override `ConvertToOnnx` on their own layer types. |

## Other constraints

- **Element type:** float32 only in v0.1. Weights and biases are down-converted from `T` via `NumOps.ToDouble` → `(float)`. Future PRs can add float64 / int64.
- **Opset version:** defaults to opset 17 (broadly supported by `onnxruntime` ≥ 1.15, Spark MLLib loaders, ML.NET). Configurable via `OnnxExportOptions.OpsetVersion`.
- **Batch axis:** declared as symbolic (`-1` → `batch_0`) so the exported graph runs at any batch size at inference time. Verified by the Telco-Churn end-to-end test.
- **Multi-target:** export works on both `net10.0` and `net471`. The round-trip tests that use `Microsoft.ML.OnnxRuntime.InferenceSession` are gated to `net10.0` because ORT's native library binding isn't stable on the net471 test host.

## Relationship to the legacy `OnnxExporter`

The hand-rolled `OnnxExporter.cs` (proof-of-concept, self-described as not production-ready) still exists in `src/Onnx/` and is unchanged. Its tests in `tests/AiDotNet.Tests/Onnx/OnnxSymbolicAxisRuntimeTests.cs` continue to pass. Once the protobuf-based path (the matrix above) reaches functional parity with the legacy exporter, the legacy code will be removed in a follow-up PR.

## How to add a new layer to this matrix

1. Add a `ConvertToOnnx` override to the layer's class. Use the existing layers (`DenseLayer.cs`, `ActivationLayer.cs`, `BatchNormalizationLayer.cs`, `DropoutLayer.cs`) as templates.
2. Add an `Onnx<LayerName>ExportTests.cs` in `tests/AiDotNet.Tests/Onnx/` covering at minimum:
   - The op type appears in the emitted bytes
   - Round-trip via `Microsoft.ML.OnnxRuntime.InferenceSession` matches AiDotNet's `Forward()` output within 1e-5 (under `#if !NET471`)
   - Any failure modes throw `OnnxExportUnsupportedException` with a useful message
3. Update this matrix's table to mark the layer ✅ with the ONNX op(s) emitted.
4. Run the full test suite to confirm no regressions.

## References

- ONNX operator reference: https://github.com/onnx/onnx/blob/v1.17.0/docs/Operators.md
- ONNX spec opset 17 release notes: https://github.com/onnx/onnx/releases/tag/v1.13.0
- AiDotNet design doc: `docs/ONNX_EXPORT_DESIGN.md`
