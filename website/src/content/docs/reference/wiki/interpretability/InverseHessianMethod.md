---
title: "InverseHessianMethod"
description: "Methods for computing inverse Hessian-vector products."
section: "API Reference"
---

`Enums` · `AiDotNet.Interpretability.Explainers`

Methods for computing inverse Hessian-vector products.

## For Beginners

The Hessian is a huge matrix of second derivatives.
We need its inverse times a vector, but computing the inverse directly is too slow.
These methods approximate H^(-1) * v efficiently.

## Fields

| Field | Summary |
|:-----|:--------|
| `ConjugateGradient` | Conjugate Gradient method. |
| `Direct` | Direct matrix inversion. |
| `LiSSA` | LiSSA (Linear-time Stochastic Second-order Algorithm). |

