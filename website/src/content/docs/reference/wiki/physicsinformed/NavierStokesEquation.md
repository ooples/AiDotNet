---
title: "NavierStokesEquation<T>"
description: "Represents the incompressible Navier-Stokes equations for fluid dynamics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the incompressible Navier-Stokes equations for fluid dynamics.

## How It Works

For Beginners:
The Navier-Stokes equations describe the motion of viscous fluids like water, air, and oil.
They are fundamental to understanding weather patterns, ocean currents, blood flow, and aerodynamics.

The equations consist of:

1. **Continuity Equation** (mass conservation): The fluid cannot be created or destroyed

∂u/∂x + ∂v/∂y = 0

2. **Momentum Equations**: Newton's second law for fluid motion

X-momentum: ∂u/∂t + u(∂u/∂x) + v(∂u/∂y) = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
Y-momentum: ∂v/∂t + u(∂v/∂x) + v(∂v/∂y) = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)

Variables:

- u(x,y,t): Velocity in x-direction
- v(x,y,t): Velocity in y-direction
- p(x,y,t): Pressure
- ν (nu): Kinematic viscosity (how "thick" the fluid is)

Physical Interpretation:

- Left side (∂u/∂t + u∂u/∂x + ...): How velocity changes following a fluid particle
- Pressure terms (-∂p/∂x, -∂p/∂y): Forces from pressure differences
- Viscous terms (ν∂²u/∂x²): Friction/drag within the fluid

Applications:

- Aircraft design (aerodynamics)
- Weather prediction
- Blood flow in arteries
- Ocean currents
- Pipe flow engineering

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NavierStokesEquation(,)` | Initializes a new instance of the Navier-Stokes equations. |
| `NavierStokesEquation(Double,Double)` | Initializes a new instance of the Navier-Stokes equations with double parameters. |

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

