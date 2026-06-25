---
title: "MetricDirection"
description: "Specifies whether higher or lower values are better for a metric."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies whether higher or lower values are better for a metric.

## For Beginners

This tells the system how to interpret a metric value:

- **Higher is better:** 0.95 accuracy is better than 0.80
- **Lower is better:** 0.05 error is better than 0.20

Knowing this is essential for model comparison, early stopping, and hyperparameter tuning.

## How It Works

Different metrics have different optimization directions:

- Accuracy, F1, AUC → Higher is better
- Error, Loss, MSE → Lower is better

## Fields

| Field | Summary |
|:-----|:--------|
| `HigherIsBetter` | Higher values indicate better performance. |
| `LowerIsBetter` | Lower values indicate better performance. |
| `NotApplicable` | Direction depends on context or is not applicable. |
| `TargetValue` | Target a specific value (e.g., calibration metrics targeting 0 or 1). |

