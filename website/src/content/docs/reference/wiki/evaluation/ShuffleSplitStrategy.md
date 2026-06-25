---
title: "ShuffleSplitStrategy<T>"
description: "Shuffle Split (Monte Carlo Cross-Validation): random train/test splits repeated multiple times."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Shuffle Split (Monte Carlo Cross-Validation): random train/test splits repeated multiple times.

## For Beginners

Shuffle Split randomly samples training and test sets:

- Unlike K-Fold, each split is independent (some samples may appear multiple times in test)
- You control exact train/test sizes
- More flexible than K-Fold - can have any test proportion
- Good for very large datasets where you want quick estimates

## How It Works

**Comparison with K-Fold:**

- K-Fold: Each sample appears in test exactly once
- Shuffle Split: Samples may appear in test 0, 1, or multiple times

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShuffleSplitStrategy(Int32,Double,Nullable<Int32>)` | Initializes Shuffle Split cross-validation. |

