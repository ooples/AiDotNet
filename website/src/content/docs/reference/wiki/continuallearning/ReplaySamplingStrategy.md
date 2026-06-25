---
title: "ReplaySamplingStrategy"
description: "Strategy for sampling during replay (training time)."
section: "API Reference"
---

`Enums` · `AiDotNet.ContinualLearning.Memory`

Strategy for sampling during replay (training time).

## Fields

| Field | Summary |
|:-----|:--------|
| `PriorityBased` | Priority-based sampling using importance weights. |
| `RecencyWeighted` | Recency-weighted sampling favors more recent examples. |
| `TaskBalanced` | Task-balanced sampling ensures equal representation of each task. |
| `Uniform` | Uniform random sampling from the buffer. |

