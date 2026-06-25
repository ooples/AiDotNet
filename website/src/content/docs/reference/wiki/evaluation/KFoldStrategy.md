---
title: "KFoldStrategy<T>"
description: "K-Fold cross-validation: splits data into K equal-sized folds, using each as validation once."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

K-Fold cross-validation: splits data into K equal-sized folds, using each as validation once.

## For Beginners

K-Fold is the most common cross-validation approach:

- Data is divided into K equal parts (folds)
- Model is trained K times, each time using a different fold for validation
- Results are averaged across all K runs

Common choices: K=5 or K=10. Higher K means more compute but lower variance in estimates.

## How It Works

**Example with K=5:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KFoldStrategy(Int32,Boolean,Nullable<Int32>)` | Initializes K-Fold cross-validation. |

