---
title: "IPredictionBasedCriterion<T, TInput, TOutput>"
description: "Interface for stopping criteria that need prediction access."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for stopping criteria that need prediction access.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeStability` | Computes the prediction stability metric. |
| `UpdatePredictions(IFullModel<,,>,IDataset<,,>)` | Updates the criterion with new predictions. |

