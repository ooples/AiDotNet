---
title: "LeavePOutStrategy<T>"
description: "Leave-P-Out cross-validation: train on N-P samples, validate on P samples for all combinations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Leave-P-Out cross-validation: train on N-P samples, validate on P samples for all combinations.

## For Beginners

Leave-P-Out is a generalization of Leave-One-Out:

- P=1: Leave-One-Out (LOO)
- P=2: Leave-Two-Out (exhaustive but expensive)
- Number of folds = C(N,P) = N!/(P!(N-P)!)
- Very thorough but computationally expensive for large datasets

## How It Works

**Warning:** The number of combinations grows rapidly. For N=20 and P=2, there are 190 folds.
For N=100 and P=2, there are 4,950 folds. Use with caution on larger datasets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeavePOutStrategy(Int32,Nullable<Int32>)` | Initializes Leave-P-Out cross-validation. |

