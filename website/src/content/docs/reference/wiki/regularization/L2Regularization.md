---
title: "L2Regularization<T, TInput, TOutput>"
description: "Implements L2 regularization (also known as Ridge), a technique that adds a penalty equal to the square of the magnitude of coefficients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regularization`

Implements L2 regularization (also known as Ridge), a technique that adds a penalty equal to the
square of the magnitude of coefficients.

## For Beginners

L2 regularization helps create smoother models by making all coefficients smaller.

Think of it like a gentle pull that shrinks all coefficients proportionally:

- It doesn't eliminate features entirely (unlike L1 regularization)
- It reduces the impact of all features by making their coefficients smaller
- It particularly penalizes large coefficient values

## How It Works

L2 regularization adds a penalty term to the loss function equal to the sum of the squared values
of the model coefficients, multiplied by a regularization strength parameter. This encourages smaller,
more evenly distributed coefficient values, which helps prevent overfitting and improves model stability.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `L2Regularization(RegularizationOptions)` | Initializes a new instance of the L2Regularization class with optional custom options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Regularize(Matrix<>)` | Applies L2 regularization to a matrix. |
| `Regularize(Vector<>)` | Applies L2 regularization to a vector. |
| `Regularize(Vector<>,Vector<>)` | Adjusts the gradient vector to account for L2 regularization during optimization. |

