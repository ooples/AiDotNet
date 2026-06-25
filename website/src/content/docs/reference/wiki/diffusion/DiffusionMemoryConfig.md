---
title: "DiffusionMemoryConfig"
description: "Configuration for diffusion model memory management."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Diffusion.Memory`

Configuration for diffusion model memory management.

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointEveryNLayers` | Number of layers between checkpoints. |
| `Default` | Creates a default configuration for single-GPU training. |
| `MaxPoolMemoryMB` | Maximum memory for activation pool in MB. |
| `MaxTensorsPerSize` | Maximum tensors per size class in the pool. |
| `MemoryEfficient` | Creates a memory-efficient configuration with aggressive checkpointing. |
| `NumDevices` | Number of devices for model sharding. |
| `ShardingStrategy` | Strategy for distributing layers across devices. |
| `SpeedOptimized` | Creates a speed-optimized configuration with less checkpointing. |
| `UseActivationPooling` | Whether to use activation pooling. |
| `UseGradientCheckpointing` | Whether to use gradient checkpointing. |
| `UsePipelineParallelism` | Whether to use pipeline parallelism for overlapping compute and transfer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `MultiGpu(Int32)` | Creates a multi-GPU configuration. |

