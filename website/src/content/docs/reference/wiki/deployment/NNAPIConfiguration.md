---
title: "NNAPIConfiguration"
description: "Configuration for NNAPI backend."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Mobile.Android`

Configuration for NNAPI backend.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowCpuFallback` | Gets or sets whether to allow fallback to CPU (default: true). |
| `AllowFp16` | Gets or sets whether to allow FP16 precision for faster inference (default: true). |
| `EnableModelCaching` | Gets or sets whether to cache compiled models (default: true). |
| `ExecutionPreference` | Gets or sets the execution preference. |
| `MaxConcurrentExecutions` | Gets or sets the maximum number of concurrent executions (default: 1). |
| `ModelCacheDirectory` | Gets or sets the model cache directory. |
| `PreferredDevice` | Gets or sets the preferred acceleration device. |
| `UseRelaxedFloat32` | Gets or sets whether to use relaxed float32 precision (default: true). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForLowPower` | Creates a configuration for low power consumption. |
| `ForMaxPerformance` | Creates a configuration for maximum performance. |

