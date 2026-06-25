---
title: "TensorRTConverter<T, TInput, TOutput>"
description: "Converts models to TensorRT optimized format for NVIDIA GPU deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.TensorRT`

Converts models to TensorRT optimized format for NVIDIA GPU deployment.
Properly integrates with IFullModel architecture.

## For Beginners

TensorRTConverter provides AI safety functionality. Default values follow the original paper settings.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertToTensorRT(IFullModel<,,>,String,TensorRTConfiguration)` | Converts a model to TensorRT format. |
| `ConvertToTensorRTBytes(IFullModel<,,>,TensorRTConfiguration)` | Converts a model to TensorRT format and returns as byte array. |
| `SerializeTensorRTEngine(TensorRTEngineBuilder,String,TensorRTConfiguration)` | Serializes TensorRT engine configuration for use with ONNX Runtime TensorRT execution provider. |

