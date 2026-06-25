---
title: "MemoryEstimate"
description: "Memory usage estimate."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Memory`

Memory usage estimate.

## Properties

| Property | Summary |
|:-----|:--------|
| `PerDeviceMemory` | Memory per device if sharded (bytes). |
| `SavingsPercent` | Percentage memory savings from checkpointing. |
| `TotalShardedMemory` | Total memory across all devices if sharded (bytes). |
| `WithCheckpointing` | Estimated memory usage with checkpointing (bytes). |
| `WithoutCheckpointing` | Estimated memory usage without checkpointing (bytes). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Gets a human-readable summary. |

