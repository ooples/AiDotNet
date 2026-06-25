---
title: "ActivationPropertyAttribute"
description: "Declares mathematical properties of an activation function for automatic test generation and cataloging."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Declares mathematical properties of an activation function for automatic test generation
and cataloging. The source generator reads these to configure invariant test parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundLower` | Lower bound of the output range for bounded activations. |
| `BoundUpper` | Upper bound of the output range for bounded activations. |
| `Cost` | Relative computational cost. |
| `HasLearnableParameters` | Whether it has learnable parameters (PReLU). |
| `IsBounded` | Whether the output is bounded. |
| `IsDifferentiable` | Whether it's differentiable everywhere. |
| `IsMonotonic` | Whether the activation is monotonically non-decreasing. |
| `IsStochastic` | Whether the activation uses randomness during training (e.g., RReLU). |
| `IsVectorActivation` | Whether this operates on vectors (Softmax) vs scalars (ReLU). |
| `ZeroPreserving` | Whether Activate(0) produces exactly 0. |

