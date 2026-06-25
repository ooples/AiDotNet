---
title: "OnnxExporter"
description: "Exports AiDotNet models to the ONNX format."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Onnx`

Exports AiDotNet models to the ONNX format.

## For Beginners

Use this class to convert your trained AiDotNet
models to ONNX format for deployment:

## How It Works

This exporter supports sequential models with common layer types:

- Dense/Linear layers (exported as MatMul + Add)
- Activation functions (ReLU, Sigmoid, Tanh, etc.)
- Dropout (exported as Identity in inference mode)

**Limitations:** This is a proof-of-concept implementation that works
with specific, known model structures (sequential models with supported layers).
For production use, consider using framework-native export tools.

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildAxisSpec(IFullModel<,,>,Int32[],Boolean)` | Builds the per-axis spec for an input or output tensor, marking axes as symbolic (`dim_param`) where the source model declared them dynamic at construction. |
| `Export(IFullModel<,,>,String,Int32[])` | Exports a model to ONNX format. |
| `ExportToBytes(IFullModel<,,>,Int32[])` | Exports a model to ONNX format and returns the bytes. |

