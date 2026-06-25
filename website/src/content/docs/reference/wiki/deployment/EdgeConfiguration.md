---
title: "EdgeConfiguration"
description: "Configuration for edge device deployment optimization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Edge`

Configuration for edge device deployment optimization.

## For Beginners

EdgeConfiguration provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheSizeMB` | Gets or sets the cache size for intermediate results in MB (default: 5 MB). |
| `EnableAdaptiveInference` | Gets or sets whether to enable adaptive inference (default: false). |
| `EnableArmNeonOptimization` | Gets or sets whether to enable ARM NEON optimization (default: true). |
| `EnableLayerFusion` | Gets or sets whether to enable layer fusion (default: true). |
| `EnableModelPartitioning` | Gets or sets whether to enable model partitioning for cloud+edge (default: false). |
| `MaxMemoryUsageMB` | Gets or sets the maximum memory usage in megabytes (default: 50 MB). |
| `MaxModelSizeMB` | Gets or sets the maximum model size in megabytes (default: 10 MB). |
| `OptimizeForPower` | Gets or sets whether to optimize for power consumption (default: true). |
| `PartitionStrategy` | Gets or sets the partition strategy. |
| `PruningRatio` | Gets or sets the pruning ratio (0.0 to 1.0, default: 0.3 = 30% sparsity). |
| `QuantizationMode` | Gets or sets the quantization mode for edge devices. |
| `TargetDevice` | Gets or sets the target device type. |
| `TargetLatencyMs` | Gets or sets the target inference latency in milliseconds (default: 100 ms). |
| `UsePruning` | Gets or sets whether to use pruning (default: true). |
| `UseQuantization` | Gets or sets whether to use quantization (default: true). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForCloudEdge` | Creates a configuration for cloud+edge partitioned deployment. |
| `ForJetson` | Creates a configuration for NVIDIA Jetson devices. |
| `ForMicrocontroller` | Creates a configuration for microcontrollers (MCU). |
| `ForRaspberryPi` | Creates a configuration for Raspberry Pi. |

