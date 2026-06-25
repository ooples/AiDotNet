---
title: "CrossValidationType"
description: "Defines the types of cross-validation strategies available."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the types of cross-validation strategies available.

## Fields

| Field | Summary |
|:-----|:--------|
| `GroupKFold` | Group K-Fold keeps samples from the same group together. |
| `KFold` | K-Fold cross-validation splits the data into k equal folds. |
| `LeaveOneOut` | Leave-One-Out uses a single sample for validation. |
| `MonteCarlo` | Monte Carlo uses repeated random sampling. |
| `Nested` | Nested cross-validation for hyperparameter tuning and evaluation. |
| `Standard` | Standard cross-validation with no special considerations. |
| `StratifiedKFold` | Stratified K-Fold maintains class distribution in each fold. |
| `TimeSeries` | Time Series cross-validation respects temporal ordering. |

