---
title: "PDEDerivatives<T>"
description: "Holds the derivatives needed for PDE computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.Interfaces`

Holds the derivatives needed for PDE computation.

## How It Works

For Beginners:
Derivatives tell us how fast a function is changing. For PDEs, we need:

- First derivatives (gradient): How fast the solution changes in each direction
- Second derivatives (Hessian): How fast the rate of change itself is changing

These are computed automatically using automatic differentiation.

## Properties

| Property | Summary |
|:-----|:--------|
| `FirstDerivatives` | First-order derivatives (gradient) of the output with respect to each input dimension. |
| `HigherDerivatives` | Higher-order derivatives (4th order and above) if needed for the specific PDE. |
| `SecondDerivatives` | Second-order derivatives (Hessian) of the output with respect to input dimensions. |
| `ThirdDerivatives` | Third-order derivatives of the output with respect to input dimensions. |

