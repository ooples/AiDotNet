# ONNX Export Design

| | |
|---|---|
| **Status** | Draft — awaiting review |
| **Scope** | Add `ExportToOnnx` to AiDotNet so trained models can be consumed outside .NET (Python, JVM, Databricks, browser). |
| **Branch** | `feature/onnx-export` |
| **Driver** | Demo for the Databricks intro session (train in AiDotNet → export to ONNX → batch inference on a Spark DataFrame in a Databricks Community Edition notebook). |

---

## Goal

Allow an `AiModelResult` produced by `AiModelBuilder` to be exported to a `.onnx` file that any ONNX-Runtime-compatible host (Python `onnxruntime`, ML.NET, browser via `onnxruntime-web`, etc.) can load and score against. The first release covers exactly the layer types the Telco Customer Churn demo needs; later releases extend coverage one layer family at a time.

## Why now

- The training session's headline demo is *"AiDotNet trains a model in .NET, scores it inside Databricks against a Delta table at scale."* Without ONNX export, the demo has to fall back to a fragile native-binary-plus-Python-loader, which materially weakens the value proposition for devs.
- Once shipped, the export path becomes generally useful — any AiDotNet user who needs cross-runtime portability gets it.

## Approach options

Three reasonable paths. Recommendation first.

### Option A — Per-layer converter + Microsoft.ML.OnnxRuntime (recommended)

Add `IOnnxConvertible` to `LayerBase`; each layer type implements `ConvertToOnnx(GraphBuilder)` to emit its operator(s). A new `AiDotNet.Onnx` namespace (probably co-located with `AiDotNet.Serving/Persistence/`) owns the graph builder, the `Microsoft.ML.OnnxRuntime` dependency, and the file-write logic. The facade exposes a single entry point on `AiModelResult`.

**Pros**
- Standard, well-supported library for the ONNX protobuf format
- Per-layer ownership means coverage grows incrementally without architectural rework
- Test pattern is simple per layer (export → load with Python `onnxruntime` → assert outputs match within tolerance)

**Cons**
- Every supported layer needs a `ConvertToOnnx` method written and tested
- Adds a NuGet dep (`Microsoft.ML.OnnxRuntime.Managed`, MIT, ~12 MB)

### Option B — TorchSharp bridge

Convert AiDotNet layer trees → TorchSharp modules → use TorchSharp's existing ONNX export.

**Pros** — Leverages PyTorch's mature exporter

**Cons** — Adds a huge dep (TorchSharp pulls in libtorch native, hundreds of MB); only covers what TorchSharp supports; the AiDotNet→TorchSharp conversion itself is a per-layer surface area very similar to Option A; doubles the runtime cost for what should be a serialization concern.

### Option C — Hand-rolled protobuf writer

Write the ONNX protobuf bytes directly with no library dependency.

**Pros** — Zero new dependencies

**Cons** — Reinvents Microsoft.ML.OnnxRuntime; brittle against ONNX spec updates; no upside over Option A.

### Recommendation: **Option A.**

Per-layer converter is the right separation. Sets up incremental rollout. NuGet dep is small, MIT-licensed (compatible with BSL 1.1), and is what every serious .NET ONNX library uses.

---

## v0.1 Scope

**Layer types supported in v0.1** — chosen to cover the Telco Customer Churn demo end-to-end. The demo is a standard tabular classifier: input feature vector → a couple of dense layers with nonlinearity → softmax.

| Layer | ONNX operator(s) | Why in v0.1 |
|---|---|---|
| `DenseLayer` | `Gemm` (or `MatMul` + `Add`) | Required for any feedforward classifier |
| `ActivationLayer` (ReLU) | `Relu` | Demo's hidden layers |
| `ActivationLayer` (Sigmoid) | `Sigmoid` | Common output activation for binary classification |
| `ActivationLayer` (Tanh) | `Tanh` | Common hidden activation |
| `ActivationLayer` (Softmax) | `Softmax` | Required for multi-class output |
| `BatchNormalizationLayer` | `BatchNormalization` | Common preprocessing layer in tabular models |
| `DropoutLayer` | `Dropout` (inference mode = identity) | Present in many trained models; export must handle gracefully even though it's a no-op at inference time |

**Out of scope for v0.1** (each will be its own follow-up PR)
- Convolutional layers (`Conv1D`, `Conv2D`, `Conv3D`, transposed convs)
- Pooling layers (`MaxPool`, `AveragePool`, `GlobalPool`)
- RNN / LSTM / GRU layers
- Attention / Transformer layers
- Embedding layers
- Custom user-defined layers
- Quantized exports
- Dynamic input shapes (v0.1 assumes fixed input shape from training data)
- Model metadata round-trip (optimizer state, training history)

---

## Facade integration

The user-facing surface stays in line with AiDotNet's strict facade rule: users only ever touch `AiModelBuilder` (to build) and `AiModelResult` (to use). ONNX export is a *use* concern, so it lives on `AiModelResult`.

### API surface

```csharp
// New extension method in AiDotNet.Serving (already has AiModelResultExtensions.cs).
namespace AiDotNet.Serving.Extensions;

public static class AiModelResultOnnxExtensions
{
    /// <summary>
    /// Exports the trained model to an ONNX file at <paramref name="filePath"/>.
    /// Throws <see cref="OnnxExportUnsupportedException"/> if the model contains
    /// a layer type not yet supported by ONNX export (see ONNX_EXPORT_DESIGN.md).
    /// </summary>
    public static void ExportToOnnx<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        string filePath,
        OnnxExportOptions? options = null);

    /// <summary>
    /// Same as <see cref="ExportToOnnx"/> but writes to a stream so callers can
    /// upload directly to cloud storage without a local file step.
    /// </summary>
    public static void ExportToOnnx<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Stream stream,
        OnnxExportOptions? options = null);

    /// <summary>
    /// Returns true if every layer in the model has an ONNX converter today.
    /// Lets callers gate UX (e.g., disable the Export button) without raising.
    /// </summary>
    public static bool CanExportToOnnx<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result);
}
```

```csharp
public sealed class OnnxExportOptions
{
    /// <summary>ONNX opset version to target. Default: 17 (broadly supported).</summary>
    public int? OpsetVersion { get; init; }

    /// <summary>Model producer name written into ONNX metadata.</summary>
    public string? ProducerName { get; init; } = "AiDotNet";

    /// <summary>Human-readable description written into ONNX metadata.</summary>
    public string? ModelDescription { get; init; }

    /// <summary>Names to assign to input tensors. If null, defaults to "input_0", "input_1", ...</summary>
    public IReadOnlyList<string>? InputNames { get; init; }

    /// <summary>Names to assign to output tensors. If null, defaults to "output_0", "output_1", ...</summary>
    public IReadOnlyList<string>? OutputNames { get; init; }
}

public sealed class OnnxExportUnsupportedException : InvalidOperationException
{
    public OnnxExportUnsupportedException(string layerTypeName, string suggestion)
        : base($"ONNX export does not yet support layer type '{layerTypeName}'. {suggestion}") { }
}
```

### `AiModelBuilder` touch points

`AiModelBuilder` itself stays untouched — building / training is independent of export. The only build-time concern is letting users opt in to an export-readiness check during `Build()`:

```csharp
// Existing partial classes get a new one:
// src/AiModelBuilder.OnnxExport.cs
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Validates at Build() time that every layer in the resulting model can be
    /// exported to ONNX. If any layer is unsupported, Build() throws instead of
    /// returning a model that would fail to export later. Optional opt-in.
    /// </summary>
    public AiModelBuilder<T, TInput, TOutput> RequireOnnxExportable()
    {
        _requireOnnxExportable = true;
        return this;
    }
}
```

### Per-layer contract

```csharp
// src/NeuralNetworks/Layers/LayerBase.cs gets a virtual method (default throws):
public abstract class LayerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Emits this layer as one or more ONNX nodes into the provided graph builder.
    /// Layers that override this method become ONNX-exportable. The default
    /// implementation throws OnnxExportUnsupportedException so users get a clear
    /// signal when they hit an unsupported layer.
    /// </summary>
    public virtual OnnxLayerOutputs ConvertToOnnx(OnnxGraphBuilder builder, OnnxLayerInputs inputs)
        => throw new OnnxExportUnsupportedException(
            GetType().Name,
            "Add a ConvertToOnnx override to this layer, or open an issue requesting it.");
}
```

`OnnxGraphBuilder` is a thin facade over `Microsoft.ML.OnnxRuntime`'s graph types. `OnnxLayerInputs` / `OnnxLayerOutputs` are records that name the tensors flowing in/out of a layer node so the next layer can wire up.

### Where files live

```
src/
  Onnx/                                    ← new namespace (NOT in Serving/, since it
    OnnxGraphBuilder.cs                      ONNX is fundamentally an artifact format,
    OnnxLayerInputs.cs                       not a serving concept; the file lives next
    OnnxLayerOutputs.cs                      to the persistence layer for now and can
    OnnxExportOptions.cs                     move if a clear home emerges)
    OnnxExportUnsupportedException.cs
  NeuralNetworks/Layers/
    DenseLayer.cs                          ← add ConvertToOnnx override
    ActivationLayer.cs                     ← add ConvertToOnnx override
    BatchNormalizationLayer.cs             ← add ConvertToOnnx override
    DropoutLayer.cs                        ← add ConvertToOnnx override
    LayerBase.cs                           ← add the virtual ConvertToOnnx
  AiModelBuilder.OnnxExport.cs             ← new partial class with RequireOnnxExportable
  AiDotNet.Serving/Extensions/
    AiModelResultOnnxExtensions.cs         ← public extension methods
```

---

## Test strategy

Per-layer round-trip test pattern:

1. **C# side** (xUnit): instantiate a layer with known weights, run a sample input forward, capture the output.
2. **Export** the layer-as-a-tiny-model to ONNX bytes.
3. **Python side** (Python script in `tests/onnx/`): load the ONNX, run the same sample input through `onnxruntime`, assert outputs match within `1e-5` tolerance.
4. **End-to-end test**: build a Telco-Churn-shaped model (2-layer dense with ReLU + sigmoid), train on a tiny synthetic dataset, export, score in Python, compare predictions row-by-row.

The Python step is run via a single helper script that the test fixture invokes with `Process.Start`. The script and the synthetic ONNX file are written to a temp directory per test so tests can run in parallel.

Coverage targets for the v0.1 PR:
- Per supported layer: unit test (in `tests/AiDotNet.Tests/Onnx/`) verifying both `ConvertToOnnx` produces a valid ONNX node and the round-trip prediction matches within tolerance.
- One end-to-end test with the demo's actual model architecture.
- Negative tests for `OnnxExportUnsupportedException`, malformed `OnnxExportOptions`, and stream-write failures.

---

## Dependencies + license compliance

| Dep | Version | License | Notes |
|---|---|---|---|
| `Microsoft.ML.OnnxRuntime.Managed` | latest stable (1.x) | MIT | Compatible with BSL 1.1 — adds a runtime dep, doesn't change AiDotNet's own license. |
| (Python test deps) `onnxruntime`, `numpy` | latest stable | MIT / BSD | Only used by the Python verification step in tests, not shipped. |

AiDotNet is BSL 1.1. The MIT-licensed ONNX Runtime is a permissive upstream dep, fine to take as a runtime dependency. No per-file license header is added (none of AiDotNet's existing files have one — the repo-level `LICENSE` covers it).

---

## Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | ONNX operator semantics differ subtly from AiDotNet's layer math (axis conventions, broadcasting rules, parameter ordering). Round-trip tests would catch this but only for the inputs they cover. | Test with multiple shaped inputs per layer (batch dims, broadcast cases). Document any documented divergences. |
| R2 | `Microsoft.ML.OnnxRuntime.Managed` only supports certain opset versions on certain runtimes. Default opset 17 may not match older Spark/Databricks runtimes. | `OnnxExportOptions.OpsetVersion` lets callers downgrade. Documentation lists tested combinations. |
| R3 | BSL 1.1 + MIT runtime dep — someone may misread the combination as relicensing. | LICENSE-of-deps section in the export doc makes the dependency-licensing story explicit. |
| R4 | Demo timing: ONNX work must finish *before* the deck's screenshot step that shows the export running. | Spec → implement → demo-first dev cycle. The Telco Churn end-to-end test in v0.1 also serves as the demo script. |
| R5 | Layer coverage grows organically and lacks a canonical "what's supported" list users can rely on. | `CanExportToOnnx` returns the boolean answer. A new `OnnxSupportMatrix.md` ships in docs and is updated every time a new layer's converter lands. |

---

## Next steps

1. **Review this spec.** Owner: Franklin. Anything to change before code starts?
2. Once approved, implement in this order:
   - `OnnxGraphBuilder` + `OnnxLayerInputs/Outputs` + `OnnxExportOptions` + `OnnxExportUnsupportedException` (no behaviour; just types)
   - `LayerBase.ConvertToOnnx` virtual + test that default throws cleanly
   - `DenseLayer.ConvertToOnnx` + round-trip test
   - `ActivationLayer.ConvertToOnnx` (ReLU, Sigmoid, Tanh, Softmax variants) + tests
   - `BatchNormalizationLayer` + `DropoutLayer` + tests
   - `AiModelResultOnnxExtensions.ExportToOnnx(filePath/stream)` + `CanExportToOnnx`
   - End-to-end Telco-Churn-shaped test
   - `AiModelBuilder.RequireOnnxExportable()` + test
   - `OnnxSupportMatrix.md` doc
3. Open a PR titled `feat(onnx): add ONNX export for v0.1 layer types (Dense / Activations / BN / Dropout)`.
4. Hand the demo team the trained model + the resulting `.onnx` file so they can put it in a Databricks notebook.

---

## References

- ONNX spec: https://onnx.ai/onnx/intro/concepts.html
- Microsoft.ML.OnnxRuntime docs: https://onnxruntime.ai/docs/api/csharp/
- AiDotNet `CHECKPOINTING_GUIDE.md` — pattern reference for how a new cross-cutting concern lands in the codebase
- AiDotNet `AiDotNet.Serving/Persistence/` — adjacent existing infrastructure to review during implementation
