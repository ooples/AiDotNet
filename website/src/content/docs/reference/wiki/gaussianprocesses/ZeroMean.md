---
title: "ZeroMean<T>"
description: "Implements a zero mean function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a zero mean function.

## For Beginners

The zero mean function returns zero for all inputs.
This is the most common default choice for GPs.

m(x) = 0

Why use zero mean?

- Simplicity: No parameters to tune
- Flexibility: The GP can still learn any function through the kernel
- Data centering: Often we center our data to have zero mean anyway

When zero mean makes sense:

- Centered/normalized data
- When you want the GP to be flexible
- When you don't have strong prior beliefs about the trend

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZeroMean` | Initializes a new zero mean function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>)` | Returns a vector of zeros for all input points. |
| `Evaluate(Vector<>)` | Returns zero for any input point. |

