---
title: "NestedCVStrategy<T>"
description: "Nested Cross-Validation: uses an inner CV loop for hyperparameter tuning and outer loop for evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Nested Cross-Validation: uses an inner CV loop for hyperparameter tuning and outer loop for evaluation.

## For Beginners

Nested CV is essential when you need to both tune hyperparameters
AND get an unbiased estimate of model performance:

- Outer loop: Evaluates final model performance (e.g., 5-fold)
- Inner loop: Selects best hyperparameters (e.g., 3-fold within each outer fold)

## How It Works

**Why is this necessary?** If you tune hyperparameters on the same data you evaluate on,
you get optimistically biased estimates. Nested CV keeps evaluation data completely separate
from hyperparameter selection.

**Structure:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NestedCVStrategy(Int32,Int32,Boolean,Nullable<Int32>)` | Initializes Nested Cross-Validation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetInnerSplits(Int32[],ReadOnlySpan<>)` | Gets inner CV splits for hyperparameter tuning within a specific outer fold's training data. |

