---
title: "SelectionStrategy<T>"
description: "Strategy for selecting initial labeled samples."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.DataPreparation.Splitting.ActiveLearning`

Strategy for selecting initial labeled samples.

## Fields

| Field | Summary |
|:-----|:--------|
| `Diverse` | Select diverse samples using clustering. |
| `Random` | Select randomly from the pool. |
| `Stratified` | Ensure class balance in initial set (requires labels). |

