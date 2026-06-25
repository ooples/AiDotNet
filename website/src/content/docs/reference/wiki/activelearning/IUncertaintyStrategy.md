---
title: "IUncertaintyStrategy<T, TInput, TOutput>"
description: "Interface for query strategies that support uncertainty-based selection."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for query strategies that support uncertainty-based selection.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeUncertainty(IFullModel<,,>,)` | Computes the uncertainty for a single sample. |
| `GetPredictionProbabilities(IFullModel<,,>,)` | Gets the predicted probabilities for a sample. |

