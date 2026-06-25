---
title: "ShardingConfig"
description: "Configuration for model sharding."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Diffusion.Memory`

Configuration for model sharding.

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomDeviceAssignments` | Custom device assignments when using ShardingStrategy.Custom. |
| `MaxMemoryPerDevice` | Memory limit per device in bytes (for memory-balanced sharding). |
| `MicroBatchCount` | Number of micro-batches for pipeline parallelism. |
| `Strategy` | Strategy for distributing layers across devices. |
| `UsePipelineParallelism` | Whether to use pipeline parallelism for overlapping compute and transfer. |

