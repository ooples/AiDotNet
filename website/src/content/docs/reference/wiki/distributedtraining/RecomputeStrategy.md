---
title: "RecomputeStrategy"
description: "Strategy for recomputing activations during the backward pass."
section: "API Reference"
---

`Enums` · `AiDotNet.DistributedTraining`

Strategy for recomputing activations during the backward pass.

## Fields

| Field | Summary |
|:-----|:--------|
| `Full` | Recompute all activations between the two nearest checkpoints during backward. |
| `None` | No recomputation. |
| `Selective` | Only recompute activations that are needed for the current backward step. |

