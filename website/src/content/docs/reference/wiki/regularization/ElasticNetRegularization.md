---
title: "ElasticNetRegularization<T, TInput, TOutput>"
description: "Implements Elastic Net regularization, a hybrid approach that combines L1 (Lasso) and L2 (Ridge) regularization techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regularization`

Implements Elastic Net regularization, a hybrid approach that combines L1 (Lasso) and L2 (Ridge) regularization techniques.

## For Beginners

Elastic Net is like having two different tools to prevent your model from becoming too complex.

Think of it like this:

- L1 (Lasso) regularization tends to completely eliminate less important features (setting them to zero)
- L2 (Ridge) regularization keeps all features but makes them smaller overall
- Elastic Net lets you blend these two approaches for the best of both worlds

## How It Works

Elastic Net regularization provides the benefits of both L1 and L2 regularization methods. It helps prevent overfitting
by penalizing large coefficient values, while also encouraging sparsity (more coefficients set to zero) when appropriate.
The L1Ratio parameter controls the balance between L1 and L2 regularization.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticNetRegularization(RegularizationOptions)` | Initializes a new instance of the ElasticNetRegularization class with optional custom options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Regularize(Matrix<>)` | Applies Elastic Net regularization to a matrix. |
| `Regularize(Vector<>)` | Applies Elastic Net regularization to a vector. |
| `Regularize(Vector<>,Vector<>)` | Adjusts the gradient vector to account for Elastic Net regularization during optimization. |

