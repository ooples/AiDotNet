---
title: "RepeatedKFoldStrategy<T>"
description: "Repeated K-Fold: runs K-Fold multiple times with different random shuffles."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Repeated K-Fold: runs K-Fold multiple times with different random shuffles.

## For Beginners

Repeated K-Fold reduces variance by averaging over multiple K-Fold runs:

- Runs standard K-Fold multiple times (e.g., 10 times)
- Each repetition uses a different random shuffle
- Provides more stable estimates than single K-Fold
- Total splits = K × Repetitions (e.g., 5-fold × 10 reps = 50 evaluations)

## How It Works

**Common configurations:**

- 5-fold × 2 repetitions (10 evaluations) - quick estimate
- 10-fold × 10 repetitions (100 evaluations) - robust estimate
- 5-fold × 10 repetitions (50 evaluations) - balanced choice

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RepeatedKFoldStrategy(Int32,Int32,Nullable<Int32>)` | Initializes Repeated K-Fold cross-validation. |

