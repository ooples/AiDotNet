---
title: "PoissonEquation<T>"
description: "Represents the Poisson Equation: ∇²u = f(x,y)"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Poisson Equation: ∇²u = f(x,y)

## How It Works

For Beginners:
The Poisson Equation is one of the most important equations in physics and engineering:

- ∇²u (Laplacian of u) = ∂²u/∂x² + ∂²u/∂y² (+ ∂²u/∂z² in 3D)
- f(x,y) is a source term (known function)

Physical Interpretation:

- Models steady-state (time-independent) phenomena
- u could represent: temperature, electric potential, pressure, concentration, etc.
- f represents sources (+) and sinks (-) in the domain

Special Case:
When f = 0, it becomes Laplace's Equation (∇²u = 0), which models equilibrium states.

Applications:

- Electrostatics: Electric potential from charge distribution
- Steady heat conduction: Temperature distribution with heat sources
- Fluid dynamics: Pressure field in incompressible flow
- Gravitational potential: From mass distribution
- Image processing: Image reconstruction and smoothing

Example:
Temperature in a metal plate with heat sources/sinks reaches a steady state
described by the Poisson equation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoissonEquation(Func<[],>,Int32)` | Initializes a new instance of the Poisson Equation. |

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

