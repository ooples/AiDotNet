---
title: "TensorRTInferenceEngine<T>"
description: "High-performance inference engine for TensorRT models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.TensorRT`

High-performance inference engine for TensorRT models.
Supports multi-stream execution and CUDA graph optimization.

## For Beginners

TensorRTInferenceEngine provides AI safety functionality. Default values follow the original paper settings.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetStatistics` | Gets inference statistics for monitoring. |
| `InferAsync([])` | Performs inference on the input data. |
| `InferBatchAsync([][])` | Performs batch inference on multiple inputs concurrently. |
| `Initialize` | Initializes the inference engine and creates execution contexts. |
| `WarmUpAsync(Int32,Int32[])` | Warms up the model by running inference on dummy data. |

