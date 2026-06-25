---
title: "GpuAccelerationConfig"
description: "Configuration for GPU-accelerated training and inference."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Engines`

Configuration for GPU-accelerated training and inference.

## For Beginners

GPU makes training 10-100x faster for large models by using your
graphics card for parallel computation. This config lets you:

- Enable/disable GPU acceleration
- Choose which GPU to use (if you have multiple)
- Control when to use GPU vs CPU based on operation size
- Enable debug logging to see what's running where

## How It Works

**Phase B: GPU Acceleration Configuration**

This configuration controls when and how GPU acceleration is used during training and inference.
The default settings work well for most desktop GPUs - just call ConfigureGpuAcceleration()
without parameters for automatic GPU detection and sensible defaults.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GpuAccelerationConfig` | Creates a configuration with default GPU settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheCompiledGraphs` | Gets or sets whether to cache compiled execution graphs (default: true). |
| `DeviceIndex` | Gets or sets the GPU device index to use if multiple GPUs are available (default: 0). |
| `DeviceType` | Gets or sets the GPU device type to use (default: Auto). |
| `EnableAutoFusion` | Gets or sets whether to enable automatic kernel fusion (default: true). |
| `EnableComputeTransferOverlap` | Gets or sets whether to enable compute/transfer overlap (default: true). |
| `EnableForInference` | Gets or sets whether to enable GPU acceleration for inference (prediction) as well as training (default: true). |
| `EnableGpuPersistence` | Gets or sets whether to enable GPU persistence for neural network weights (default: true). |
| `EnableGraphCompilation` | Gets or sets whether to enable execution graph compilation (default: true). |
| `EnablePrefetch` | Gets or sets whether to enable data prefetching (default: true). |
| `EnableProfiling` | Gets or sets whether to enable GPU profiling (default: false). |
| `ExecutionMode` | Gets or sets the GPU execution mode (default: Auto). |
| `MaxComputeStreams` | Gets or sets the maximum number of compute streams (default: 3). |
| `MaxGpuMemoryUsage` | Gets or sets the maximum GPU memory usage fraction (default: 0.8). |
| `MinGpuElements` | Gets or sets the minimum number of elements to use GPU (default: 4096). |
| `TransferStreams` | Gets or sets the number of transfer streams (default: 2). |
| `UsageLevel` | Gets or sets the GPU usage level (default: Default). |
| `VerboseLogging` | Gets or sets whether to enable verbose logging of GPU operations (default: false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToExecutionOptions` | Converts this user-facing configuration to internal GpuExecutionOptions. |
| `ToString` | Gets a string representation of this configuration. |

