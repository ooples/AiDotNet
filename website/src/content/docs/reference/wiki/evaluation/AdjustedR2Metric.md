---
title: "AdjustedR2Metric<T>"
description: "Computes Adjusted R² Score: R² adjusted for the number of predictors in the model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Adjusted R² Score: R² adjusted for the number of predictors in the model.

## For Beginners

Adjusted R² penalizes adding unnecessary predictors to a model.
Unlike regular R² which always increases (or stays the same) when adding features,
adjusted R² will decrease if the new feature doesn't improve the model enough.

- Use when comparing models with different numbers of features
- Lower than R² when predictors don't contribute meaningfully
- Can be negative if the model performs very poorly

## How It Works

Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdjustedR2Metric(Int32)` | Initializes the Adjusted R² metric. |

