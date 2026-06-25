---
title: "TensorRTConfiguration"
description: "Configuration for TensorRT model conversion and execution."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.TensorRT`

Configuration for TensorRT model conversion and execution.

## Properties

| Property | Summary |
|:-----|:--------|
| `BuilderOptimizationLevel` | Gets or sets the builder optimization level (0-5, higher is more optimization). |
| `CalibrationDataPath` | Gets or sets the path to calibration data for INT8 quantization. |
| `CleanupIntermediateFiles` | Gets or sets whether to cleanup intermediate files (ONNX, etc.) (default: true). |
| `CustomPluginPaths` | Gets or sets custom plugin library paths. |
| `DLACore` | Gets or sets the DLA core to use when EnableDLA is true (-1 = disabled, 0+ = specific core). |
| `DeviceId` | Gets or sets the GPU device ID to use (default: 0). |
| `EnableCudaGraphs` | Gets or sets whether to enable CUDA graph capture (default: false). |
| `EnableDLA` | Gets or sets whether to enable Deep Learning Accelerator (DLA) on Jetson devices. |
| `EnableDynamicShapes` | Gets or sets whether to enable dynamic shapes (default: false). |
| `EnableMultiStream` | Gets or sets whether to use multi-stream execution (default: false). |
| `EnableProfiling` | Gets or sets whether to enable profiling (default: false). |
| `EngineCachePath` | Gets or sets the engine cache path for faster reloading. |
| `MaxBatchSize` | Gets or sets the maximum batch size for inference (default: 1). |
| `MaxWorkspaceSize` | Gets or sets the maximum workspace size in bytes for TensorRT (default: 1GB). |
| `NumStreams` | Gets or sets the number of streams for multi-stream execution (default: 2). |
| `OptimizationProfiles` | Gets or sets optimization profiles for dynamic shapes. |
| `Precision` | Gets or sets the inference precision mode (default: FP32). |
| `StrictTypeConstraints` | Gets or sets whether to enable strict type constraints (default: false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForHighThroughput(Int32,String)` | Creates a configuration optimized for high throughput. |
| `ForInt8(String)` | Creates a configuration with INT8 quantization. |
| `ForLowLatency` | Creates a configuration optimized for low latency (batch size 1). |
| `ForMaxPerformance` | Creates a configuration optimized for maximum performance. |
| `Validate` | Validates the configuration and throws exceptions for invalid settings. |

