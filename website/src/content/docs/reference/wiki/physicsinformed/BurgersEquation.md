---
title: "BurgersEquation<T>"
description: "Represents the Burgers' Equation: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Burgers' Equation: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²

## How It Works

For Beginners:
Burgers' Equation is a fundamental PDE that combines:

1. Nonlinear convection (u * ∂u/∂x): The solution advects (moves) at its own speed
2. Diffusion (ν * ∂²u/∂x²): The solution spreads out over time

Physical Interpretation:

- Models simplified fluid dynamics (1D version of Navier-Stokes)
- u(x,t) can represent fluid velocity at position x and time t
- ν (nu) is the viscosity - controls how much the solution smooths out
- The nonlinear term creates shock waves and turbulence-like behavior

Key Feature:
The nonlinearity makes this equation challenging - it can develop discontinuities (shocks)
even from smooth initial conditions. This makes it a perfect benchmark for PINNs.

Applications:

- Gas dynamics
- Traffic flow modeling
- Shock wave formation
- Turbulence studies

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BurgersEquation()` | Initializes a new instance of Burgers' Equation. |
| `BurgersEquation(Double)` | Initializes a new instance of Burgers' Equation with double parameter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDimension` |  |
| `Name` |  |
| `OutputDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeResidual(Vector<>,Vector<>,PDEDerivatives<>)` |  |
| `ComputeResidualGradient(Vector<>,Vector<>,PDEDerivatives<>)` |  |

