---
title: "OnnxExportOptions"
description: "Optional configuration for ONNX export."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Onnx`

Optional configuration for ONNX export. Defaults are chosen so a caller can pass
`null` and get a working file for the most common case (float32 inputs, recent
opset, broadly compatible with downstream runtimes including Databricks / Spark).

## Properties

| Property | Summary |
|:-----|:--------|
| `InputNames` | Names to assign to input tensors. |
| `ModelDescription` | Human-readable description written into the .onnx file's metadata. |
| `OpsetVersion` | ONNX opset version to target. |
| `OutputNames` | Names to assign to output tensors. |
| `ProducerName` | Producer name written into the .onnx file's metadata. |
| `ProducerVersion` | Producer version written into the .onnx file's metadata. |

