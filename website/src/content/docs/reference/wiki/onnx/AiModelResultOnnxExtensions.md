---
title: "AiModelResultOnnxExtensions"
description: "Public-facing ONNX export API on `AiModelResult`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Onnx`

Public-facing ONNX export API on `AiModelResult`.
These are the methods user code calls; everything in `src/Onnx/` below
this surface is implementation detail.

Usage:

v0.1 supports sequential models composed of: DenseLayer, ActivationLayer
(ReLU/Sigmoid/Tanh/Softmax/Identity), BatchNormalizationLayer, DropoutLayer.
Other layer types throw `OnnxExportUnsupportedException` with
the unsupported layer's type name.

## Methods

| Method | Summary |
|:-----|:--------|
| `CanExportToOnnx(AiModelResult<,,>)` | Returns true if every layer in the model has an ONNX converter today. |
| `ExportToOnnx(AiModelResult<,,>,Stream,OnnxExportOptions)` | Exports to a stream — useful for uploading directly to cloud storage. |
| `ExportToOnnx(AiModelResult<,,>,String,OnnxExportOptions)` | Exports a trained model to an ONNX file at `filePath`. |

