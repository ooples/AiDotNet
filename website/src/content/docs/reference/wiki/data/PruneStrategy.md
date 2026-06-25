---
title: "PruneStrategy"
description: "Strategy for selecting samples to prune."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Quality`

Strategy for selecting samples to prune.

## Fields

| Field | Summary |
|:-----|:--------|
| `HighConfidence` | Remove samples with consistently high confidence (easy examples). |
| `LowEL2N` | Remove samples with lowest EL2N (error L2 norm) scores. |
| `LowGraNd` | Remove samples with lowest GraNd (gradient norm) scores. |
| `NeverForgotten` | Remove samples never forgotten during training (already well-learned). |

