---
title: "ImportanceAccumulationMode"
description: "Mode for accumulating importance across tasks."
section: "API Reference"
---

`Enums` · `AiDotNet.ContinualLearning.Strategies`

Mode for accumulating importance across tasks.

## Fields

| Field | Summary |
|:-----|:--------|
| `Max` | Keep maximum importance value (most conservative). |
| `Sum` | Sum importance values across all tasks. |
| `WeightedSum` | Exponentially weighted sum favoring recent tasks. |

