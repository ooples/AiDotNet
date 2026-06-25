---
title: "LSTDOptions<T>"
description: "Configuration options for LSTD (Least-Squares Temporal Difference) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for LSTD (Least-Squares Temporal Difference) agents.

## For Beginners

LSTD is like solving a math equation directly instead of guessing and checking.
It collects experiences and then computes the best weights all at once using
linear algebra, rather than slowly adjusting them one step at a time.

Best for:

- Limited data scenarios (sample efficient)
- Batch learning from fixed datasets
- When you have computational power for matrix operations
- Problems where convergence speed matters

Not suitable for:

- Very large feature spaces (matrix becomes huge)
- Online learning (needs batches)
- When computational resources are limited
- Non-linear function approximation needs

## How It Works

LSTD solves for the optimal linear weights directly using matrix operations
(A^-1 * b) rather than incremental updates. This provides more sample-efficient
learning but requires solving a linear system.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `FeatureSize` | Number of features in the state representation. |
| `RegularizationParam` | Regularization parameter to prevent overfitting and ensure numerical stability. |

