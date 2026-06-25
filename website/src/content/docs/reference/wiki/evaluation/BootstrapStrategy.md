---
title: "BootstrapStrategy<T>"
description: "Bootstrap Cross-Validation: uses bootstrap sampling (sampling with replacement) for validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Bootstrap Cross-Validation: uses bootstrap sampling (sampling with replacement) for validation.

## For Beginners

Bootstrap creates training sets by sampling with replacement:

- Each bootstrap sample is the same size as the original data
- Some samples appear multiple times, others not at all
- Out-of-bag (OOB) samples (not selected) form the validation set
- On average, about 63.2% of samples appear in each bootstrap, 36.8% are OOB

## How It Works

**Advantages:**

- Can generate unlimited training sets
- Works well with very small datasets
- Provides good variance estimates

**Variants:**

- .632 Bootstrap: Weighted average of training and OOB error
- .632+ Bootstrap: Adds correction for overfitting

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BootstrapStrategy(Int32,Nullable<Int32>)` | Initializes Bootstrap cross-validation. |

