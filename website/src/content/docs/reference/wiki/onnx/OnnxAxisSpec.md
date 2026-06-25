---
title: "OnnxAxisSpec"
description: "Per-axis shape descriptor for an ONNX `TensorShapeProto.Dimension`."
section: "API Reference"
---

`Structs` · `AiDotNet.Onnx`

Per-axis shape descriptor for an ONNX `TensorShapeProto.Dimension`.
Encodes either a concrete `dim_value` (e.g. channel count = 3) or a
symbolic `dim_param` (e.g. `"batch"`, `"H"`, `"W"`).

## How It Works

Symbolic axes (issue #1211) let a single exported ONNX file run at any
(batch, height, width) the downstream runtime feeds it. ONNX Runtime,
OpenVINO and TensorRT all expect symbolic axes for production deployment;
PyTorch surfaces them via `torch.onnx.export(..., dynamic_axes=...)`.

Construct with the static `Int32)` or
`String)` factory methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `FixedDim` | Concrete dimension size when `SymbolicName` is null. |
| `SymbolicName` | Symbolic name (e.g. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fixed(Int32)` | Concrete-size axis (encoded as ONNX `dim_value`). |
| `Symbolic(String)` | Symbolic axis (encoded as ONNX `dim_param`). |

