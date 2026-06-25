---
title: "OnnxTensorConverter"
description: "Converts between AiDotNet Tensor types and ONNX Runtime tensor types."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Onnx`

Converts between AiDotNet Tensor types and ONNX Runtime tensor types.

## For Beginners

When running ONNX models, we need to convert our data
to a format that ONNX Runtime understands. This converter handles that translation:

- ToOnnx: Converts your AiDotNet tensor to ONNX format for model input
- FromOnnx: Converts ONNX model output back to AiDotNet tensor

## How It Works

This class provides static methods for converting tensors between AiDotNet's
generic Tensor<T> format and ONNX Runtime's DenseTensor format.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromOnnxDouble(Tensor<Double>)` | Converts an ONNX DenseTensor of doubles to an AiDotNet Tensor. |
| `FromOnnxFloat(Tensor<Single>)` | Converts an ONNX DenseTensor of floats to an AiDotNet Tensor. |
| `FromOnnxFloatRemoveBatch(Tensor<Single>)` | Converts an ONNX tensor and removes the batch dimension if it's 1. |
| `FromOnnxInt(Tensor<Int32>)` | Converts an ONNX DenseTensor of integers to an AiDotNet Tensor. |
| `FromOnnxLong(Tensor<Int64>)` | Converts an ONNX DenseTensor of long integers to an AiDotNet Tensor. |
| `FromOnnxValue(DisposableNamedOnnxValue)` | Converts an ONNX DisposableNamedOnnxValue to an AiDotNet Tensor based on its actual element type. |
| `GetOnnxTypeName(Type)` | Gets the ONNX element type name for a .NET type. |
| `ToOnnxDouble(Tensor<>)` | Converts an AiDotNet Tensor to an ONNX DenseTensor of doubles. |
| `ToOnnxFloat(Tensor<>)` | Converts an AiDotNet Tensor to an ONNX DenseTensor of floats. |
| `ToOnnxFloatWithBatch(Tensor<>)` | Creates an ONNX DenseTensor with a prepended batch dimension. |
| `ToOnnxInt(Tensor<>)` | Converts an AiDotNet Tensor to an ONNX DenseTensor of integers. |
| `ToOnnxLong(Tensor<>)` | Converts an AiDotNet Tensor to an ONNX DenseTensor of long integers. |
| `ToOnnxValue(String,Tensor<>,String)` | Converts an AiDotNet Tensor to an ONNX NamedOnnxValue based on the target element type. |

