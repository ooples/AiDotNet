---
title: "L1Regularization<T, TInput, TOutput>"
description: "Implements L1 regularization (also known as Lasso), a technique that adds a penalty equal to the absolute value of the magnitude of coefficients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regularization`

Implements L1 regularization (also known as Lasso), a technique that adds a penalty equal to the
absolute value of the magnitude of coefficients.

## For Beginners

L1 regularization helps create simpler models by completely removing less important features.

Think of it like a strict budget committee:

- It forces the model to focus only on the most important features
- Less important features get their coefficients reduced to exactly zero
- This means some features are completely eliminated from the model

## How It Works

L1 regularization adds a penalty term to the loss function equal to the sum of the absolute values
of the model coefficients, multiplied by a regularization strength parameter. This encourages sparse
models by driving some coefficients to exactly zero, effectively performing feature selection.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `L1Regularization(RegularizationOptions)` | Initializes a new instance of the L1Regularization class with optional custom options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Regularize(Matrix<>)` | Applies L1 regularization to a matrix. |
| `Regularize(Vector<>)` | Applies L1 regularization to a vector. |
| `Regularize(Vector<>,Vector<>)` | Adjusts the gradient vector to account for L1 regularization during optimization. |

