---
title: "HeatEquation<T>"
description: "Represents the Heat Equation (or Diffusion Equation): ∂u/∂t = α ∂²u/∂x²"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Heat Equation (or Diffusion Equation): ∂u/∂t = α ∂²u/∂x²

## How It Works

For Beginners:
The Heat Equation models how heat diffuses through a material over time.

- u(x,t) is the temperature at position x and time t
- α (alpha) is the thermal diffusivity, which controls how fast heat spreads
- The equation says: The rate of temperature change equals how curved the temperature profile is

Physical Interpretation:

- If the temperature profile is concave (curves down), heat flows in → temperature increases
- If the temperature profile is convex (curves up), heat flows out → temperature decreases
- At inflection points (no curvature), temperature stays constant

Example: A metal rod with one end heated - the heat gradually spreads along the rod.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeatEquation()` | Initializes a new instance of the Heat Equation. |
| `HeatEquation(Double)` | Initializes a new instance of the Heat Equation with double parameter. |

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

