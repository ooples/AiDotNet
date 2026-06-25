---
title: "BlockedKFoldStrategy<T>"
description: "Blocked K-Fold: K-Fold with temporal blocking (gap) between train and validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Blocked K-Fold: K-Fold with temporal blocking (gap) between train and validation.

## For Beginners

Blocked K-Fold adds a gap between training and validation:

- Prevents data leakage from temporal correlation
- Important for time-series-like data in standard K-Fold
- The "gap" samples are excluded from both train and validation

## How It Works

**Use case:** When data has temporal ordering but you want K-Fold style validation:

- Financial features computed from rolling windows
- User behavior data with session effects
- Any data where adjacent samples are correlated

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlockedKFoldStrategy(Int32,Int32,Boolean,Nullable<Int32>)` | Initializes Blocked K-Fold. |

