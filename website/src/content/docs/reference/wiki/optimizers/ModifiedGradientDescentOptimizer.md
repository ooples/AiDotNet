---
title: "ModifiedGradientDescentOptimizer<T>"
description: "Modified Gradient Descent optimizer for Hope architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Modified Gradient Descent optimizer for Hope architecture.
Based on Equations 27-29 from "Nested Learning" paper.

Traditional GD: W_{t+1} = W_t - η * ∇L(W_t; x_t) ⊗ x_t
Modified GD: W_{t+1} = W_t * (I - x_t*x_t^T) - η * ∇L(W_t; x_t) ⊗ x_t

This formulation uses L2 regression objective instead of dot-product similarity,
resulting in better handling of data dependencies in token space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModifiedGradientDescentOptimizer(,IEngine)` | Creates a modified gradient descent optimizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets the learning rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeIdentityMinusOuterProduct(Vector<>)` | Computes (I - x_t*x_t^T) where x_t is the input vector. |
| `ComputeOuterProduct(Vector<>,Vector<>)` | Computes outer product of two vectors: a ⊗ b = a*b^T |
| `UpdateMatrix(Matrix<>,Vector<>,Vector<>)` | Updates parameters using modified gradient descent (Equations 27-29). |
| `UpdateVector(Vector<>,Vector<>,Vector<>)` | Updates a parameter vector using modified gradient descent. |

