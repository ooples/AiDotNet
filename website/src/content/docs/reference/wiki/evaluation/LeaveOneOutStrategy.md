---
title: "LeaveOneOutStrategy<T>"
description: "Leave-One-Out Cross-Validation (LOOCV): each sample is used once as validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Leave-One-Out Cross-Validation (LOOCV): each sample is used once as validation.

## For Beginners

LOOCV is the extreme case of K-Fold where K equals the number of samples:

- Each sample gets a turn as the single validation point
- Maximizes training data usage (N-1 samples for training)
- Provides nearly unbiased estimate of model performance
- Computationally expensive: requires N model trainings

## How It Works

**When to use:**

- Very small datasets where you can't afford to hold out much data
- When computational cost is not a concern
- When you need the most accurate estimate possible

**Caution:** For datasets with N > 1000, consider using K-Fold instead.
LOOCV can also have high variance in some scenarios.

