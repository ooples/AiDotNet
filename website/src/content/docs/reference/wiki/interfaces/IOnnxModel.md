---
title: "IOnnxModel<T>"
description: "Defines the contract for ONNX model wrappers that provide cross-platform model inference."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for ONNX model wrappers that provide cross-platform model inference.

## For Beginners

ONNX (Open Neural Network Exchange) is a universal format
for neural network models. This interface allows you to:

- Load models trained in PyTorch, TensorFlow, or other frameworks
- Run inference using CPU, GPU (CUDA), TensorRT, or DirectML
- Convert between AiDotNet tensors and ONNX tensors automatically

## How It Works

This interface provides a unified way to work with ONNX models in AiDotNet.
It supports loading models from files, byte arrays, or URLs, and provides
methods for running inference with AiDotNet Tensor types.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExecutionProvider` | Gets the execution provider currently being used (CPU, CUDA, TensorRT, DirectML). |
| `IsLoaded` | Gets whether the model has been successfully loaded and is ready for inference. |
| `Metadata` | Gets the metadata about the loaded ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Run(IReadOnlyDictionary<String,Tensor<>>)` | Runs inference with named inputs. |
| `Run(Tensor<>)` | Runs inference with a single input tensor. |
| `RunAsync(IReadOnlyDictionary<String,Tensor<>>,CancellationToken)` | Runs inference asynchronously with named inputs. |
| `RunAsync(Tensor<>,CancellationToken)` | Runs inference asynchronously with a single input tensor. |
| `WarmUp` | Warms up the model by running a single inference with dummy data. |
| `WarmUpAsync(CancellationToken)` | Warms up the model asynchronously. |

