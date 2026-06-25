---
title: "IOutputDerivative<T>"
description: "Interface for activation functions that can compute their derivative given the post-activation output value rather than the pre-activation input."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for activation functions that can compute their derivative
given the post-activation output value rather than the pre-activation input.
This avoids the common bug where the derivative re-applies the activation
(e.g., sigmoid(sigmoid(x)) instead of sigmoid(x)*(1-sigmoid(x))).

## Methods

| Method | Summary |
|:-----|:--------|
| `DerivativeFromOutput(Tensor<>)` | Computes the activation derivative given the post-activation output value. |

