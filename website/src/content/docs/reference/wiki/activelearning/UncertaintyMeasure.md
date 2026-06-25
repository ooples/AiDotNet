---
title: "UncertaintyMeasure"
description: "Methods for measuring uncertainty in predictions."
section: "API Reference"
---

`Enums` · `AiDotNet.ActiveLearning.Config`

Methods for measuring uncertainty in predictions.

## Fields

| Field | Summary |
|:-----|:--------|
| `Entropy` | Shannon entropy of the predicted probability distribution. |
| `LeastConfidence` | One minus the maximum predicted probability. |
| `Margin` | Difference between top two predicted probabilities. |
| `PredictiveVariance` | Variance of predictions under MC Dropout. |

