---
title: "ConditionNumberMethod"
description: "Specifies different methods for calculating the condition number of a matrix."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different methods for calculating the condition number of a matrix.

## For Beginners

The condition number is a measure of how sensitive a matrix is to changes in input.

Think of it like checking how stable a table is - a table with uneven legs (high condition number) 
will wobble a lot with small changes, while a stable table (low condition number) remains steady.

In machine learning and numerical computing:

- A low condition number (close to 1) indicates a "well-conditioned" matrix that produces reliable results
- A high condition number indicates an "ill-conditioned" matrix that might amplify small errors
- An infinite condition number means the matrix is singular (cannot be inverted)

This is important because ill-conditioned matrices can cause numerical problems in algorithms,
leading to inaccurate results or slow convergence.

## Fields

| Field | Summary |
|:-----|:--------|
| `InfinityNorm` | Calculates the condition number using the Infinity norm (maximum absolute row sum). |
| `L1Norm` | Calculates the condition number using the L1 norm (sum of absolute column values). |
| `PowerIteration` | Estimates the condition number using the power iteration method. |
| `SVD` | Calculates the condition number using Singular Value Decomposition. |

