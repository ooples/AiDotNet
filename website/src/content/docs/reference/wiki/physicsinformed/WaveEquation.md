---
title: "WaveEquation<T>"
description: "Represents the Wave Equation: ∂²u/∂t² = c² * ∇²u"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Wave Equation: ∂²u/∂t² = c² * ∇²u

## How It Works

For Beginners:
The Wave Equation describes how waves propagate through a medium:

- u(x,t) is the wave amplitude (displacement) at position x and time t
- c is the wave speed (how fast disturbances propagate)
- ∇²u is the Laplacian (spatial curvature) of u

Physical Interpretation:

- The acceleration (∂²u/∂t²) is proportional to the spatial curvature
- If the wave is curved upward, it accelerates upward (and vice versa)
- This creates oscillations that propagate at speed c

Key Properties:

- Solutions are superpositions of traveling waves: u(x±ct)
- Energy is conserved
- Waves can reflect, refract, diffract, and interfere

Applications:

- Sound waves in air/water
- Vibrating strings (guitar, piano)
- Electromagnetic waves (light, radio)
- Seismic waves (earthquakes)
- Water waves (ocean, lake)

Example:
A guitar string vibrates according to the wave equation when plucked.
The wave speed depends on the string's tension and mass.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveEquation(,Int32)` | Initializes a new instance of the Wave Equation. |
| `WaveEquation(Double,Int32)` | Initializes a new instance of the Wave Equation with double parameters. |

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

