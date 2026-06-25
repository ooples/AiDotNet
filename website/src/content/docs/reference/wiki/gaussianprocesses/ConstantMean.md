---
title: "ConstantMean<T>"
description: "Implements a constant mean function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a constant mean function.

## For Beginners

The constant mean function returns the same value for all inputs.

m(x) = c

Why use constant mean?

- When your data has a known average value
- When you expect predictions far from training data to revert to this constant
- As a simple baseline trend

The constant can be:

- Set manually based on domain knowledge
- Learned during hyperparameter optimization
- Set to the empirical mean of your training targets

Example: If predicting house prices in thousands, you might set c = 300
to represent your prior belief that houses cost around $300k on average.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstantMean()` | Initializes a new constant mean function. |
| `ConstantMean(Double)` | Initializes a constant mean with a double value. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Constant` | Gets the constant value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>)` | Returns a vector of the constant for all input points. |
| `Evaluate(Vector<>)` | Returns the constant for any input point. |

