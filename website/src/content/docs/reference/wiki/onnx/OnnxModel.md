---
title: "OnnxModel<T>"
description: "A wrapper for ONNX models that provides easy-to-use inference with AiDotNet Tensor types."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx`

A wrapper for ONNX models that provides easy-to-use inference with AiDotNet Tensor types.

## For Beginners

Use this class to run pre-trained ONNX models:

## How It Works

This class wraps the ONNX Runtime InferenceSession and provides:

- Automatic tensor conversion between AiDotNet and ONNX formats
- Support for multiple execution providers (CPU, CUDA, TensorRT, DirectML)
- Multi-input/multi-output model support
- Warm-up and async inference

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnnxModel(Byte[],OnnxModelOptions)` | Creates a new OnnxModel from a byte array. |
| `OnnxModel(String,OnnxModelOptions)` | Creates a new OnnxModel from a file path. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExecutionProvider` |  |
| `IsLoaded` |  |
| `Metadata` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(String,OnnxModelOptions,IProgress<Double>,CancellationToken)` | Creates an OnnxModel asynchronously, optionally downloading from a URL. |
| `Dispose` | Disposes the ONNX session and releases resources. |
| `Dispose(Boolean)` | Disposes managed and unmanaged resources. |
| `Run(IReadOnlyDictionary<String,Tensor<>>)` |  |
| `Run(IReadOnlyDictionary<String,Tensor<>>,IEnumerable<String>)` | Runs inference with specific output names. |
| `Run(Tensor<>)` |  |
| `RunAsync(IReadOnlyDictionary<String,Tensor<>>,CancellationToken)` |  |
| `RunAsync(Tensor<>,CancellationToken)` |  |
| `RunWithLongInput(String,Int64[])` | Runs inference with long integer inputs (useful for token IDs). |
| `WarmUp` |  |
| `WarmUpAsync(CancellationToken)` |  |

