---
title: "OnnxModelOptions"
description: "Configuration options for loading and running ONNX models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Onnx`

Configuration options for loading and running ONNX models.

## For Beginners

Use defaults for most cases:

For GPU acceleration:

## How It Works

This class provides comprehensive configuration for ONNX model inference,
including execution provider selection, memory optimization, and threading.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnnxModelOptions` | Initializes a new instance with default values. |
| `OnnxModelOptions(OnnxModelOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoWarmUp` | Gets or sets whether to automatically warm up the model after loading. |
| `CudaMemoryLimit` | Gets or sets the CUDA memory limit in bytes (0 = no limit). |
| `CudaUseArena` | Gets or sets whether to use CUDA memory arena for better performance. |
| `CustomOptions` | Gets or sets custom session options as key-value pairs. |
| `EnableMemoryArena` | Gets or sets whether to enable memory arena. |
| `EnableMemoryPattern` | Gets or sets whether to enable memory pattern optimization. |
| `EnableProfiling` | Gets or sets whether to enable profiling for performance analysis. |
| `ExecutionProvider` | Gets or sets the preferred execution provider. |
| `FallbackProviders` | Gets or sets fallback execution providers if the primary fails. |
| `GpuDeviceId` | Gets or sets the GPU device ID for CUDA/TensorRT/DirectML providers. |
| `InterOpNumThreads` | Gets or sets the number of threads for parallel operations. |
| `IntraOpNumThreads` | Gets or sets the number of threads for CPU execution. |
| `LogLevel` | Gets or sets the log severity level for ONNX Runtime. |
| `OptimizationLevel` | Gets or sets the graph optimization level. |
| `ProfileOutputPath` | Gets or sets the path for saving profiling output. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForCpu(Nullable<Int32>)` | Creates default options for CPU execution. |
| `ForCuda(Int32)` | Creates default options for CUDA GPU execution. |
| `ForDirectML(Int32)` | Creates default options for DirectML GPU execution (Windows). |
| `ForTensorRT(Int32)` | Creates default options for TensorRT execution (NVIDIA optimized). |

