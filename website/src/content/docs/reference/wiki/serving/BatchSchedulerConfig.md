---
title: "BatchSchedulerConfig"
description: "Configuration for the batch scheduler."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Serving.ContinuousBatching`

Configuration for the batch scheduler.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowPreemption` | Whether to allow preempting lower-priority sequences. |
| `HeadDimension` | Dimension of each attention head (for memory estimation). |
| `MaxBatchSize` | Maximum number of sequences in a batch. |
| `MaxCacheSlots` | Maximum number of KV-cache slots available. |
| `MaxMemoryBytes` | Maximum memory available for KV-cache (bytes). |
| `NumHeads` | Number of attention heads (for memory estimation). |
| `NumLayers` | Number of transformer layers (for memory estimation). |
| `Policy` | Scheduling policy to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForModel(String,Int32)` | Creates config for a specific model. |

