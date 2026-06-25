---
title: "CombinationMethod<T>"
description: "Defines methods for combining strategy scores."
section: "API Reference"
---

`Enums` · `AiDotNet.ActiveLearning`

Defines methods for combining strategy scores.

## Fields

| Field | Summary |
|:-----|:--------|
| `Maximum` | Maximum score across strategies. |
| `Minimum` | Minimum score across strategies (most conservative). |
| `Product` | Product of scores (samples must be good in all strategies). |
| `RankFusion` | Rank-based fusion (combines rankings, not scores). |
| `WeightedSum` | Weighted sum of normalized scores. |

