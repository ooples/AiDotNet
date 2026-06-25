---
title: "MonteCarloStrategy<T>"
description: "Monte Carlo cross-validation (repeated random sub-sampling validation)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Monte Carlo cross-validation (repeated random sub-sampling validation).

## For Beginners

Monte Carlo CV randomly splits data multiple times:

- Each iteration creates a random train/test split
- Unlike K-Fold, samples may appear in validation multiple times or never
- More flexible control over train/test sizes
- Good for small datasets where K-Fold has high variance

## How It Works

**Comparison to K-Fold:**

- K-Fold: Each sample in validation exactly once
- Monte Carlo: Samples may repeat or be missing in validation
- Monte Carlo: More iterations possible with less computation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MonteCarloStrategy(Int32,Double,Nullable<Int32>)` | Initializes Monte Carlo cross-validation. |

