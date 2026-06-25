---
title: "NNAPIBackend<T>"
description: "NNAPI (Neural Networks API) backend for Android deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Mobile.Android`

NNAPI (Neural Networks API) backend for Android deployment.
Provides hardware acceleration on Android devices when libneuralnetworks.so
is loadable; falls back to managed CPU execution otherwise.

## For Beginners

NNAPI is the Android-side API that exposes hardware
accelerators (GPU / DSP / NPU) for neural network inference. This backend
binds to the native `libneuralnetworks.so` via P/Invoke; if the library
is not present, the backend reports unavailability and routes execution
through a managed CPU fallback when one has been configured.

## How It Works

Default values follow the Android NNAPI guidance: relaxed FP32, optional
FP16, and CPU fallback support. The `CpuExecutor` hook lets
callers plug in a real managed inference path for fallback execution.

## Properties

| Property | Summary |
|:-----|:--------|
| `CpuExecutor` | Managed-CPU inference hook invoked when NNAPI is unavailable or no compiled native graph is available. |
| `GraphBuilder` | Optional graph builder that decodes loaded model bytes into NNAPI operands and operations before compilation. |
| `OutputElementCount` | Output element count set by an upstream model-format adapter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Execute([])` | Executes inference using NNAPI or the managed CPU fallback. |
| `ExecuteAsync([])` | Executes inference asynchronously. |
| `GetPerformanceInfo` | Gets NNAPI performance information for the current device. |
| `GetSupportedDevices` | Gets the supported acceleration devices on this Android device. |
| `Initialize` | Initializes the NNAPI backend. |
| `IsNNAPIAvailable` | Checks if NNAPI is available on the current device. |
| `LoadModel(String)` | Loads a model for NNAPI execution. |

