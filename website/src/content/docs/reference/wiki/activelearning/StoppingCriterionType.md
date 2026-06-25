---
title: "StoppingCriterionType"
description: "Criteria for early stopping in active learning."
section: "API Reference"
---

`Enums` · `AiDotNet.ActiveLearning.Config`

Criteria for early stopping in active learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `BudgetExhausted` | Stop when labeling budget is exhausted. |
| `ContradictingInformation` | Stop when new labels contradict previous learning. |
| `ConvergenceDetected` | Stop when model performance converges. |
| `PerformancePlateau` | Stop when accuracy improvement plateaus. |
| `StabilizingPredictions` | Stop when predictions stabilize across iterations. |

