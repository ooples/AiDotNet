---
title: "KortewegDeVriesEquation<T>"
description: "Represents the Korteweg-de Vries (KdV) Equation: ∂u/∂t + αu∂u/∂x + β∂³u/∂x³ = 0"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Korteweg-de Vries (KdV) Equation:
∂u/∂t + αu∂u/∂x + β∂³u/∂x³ = 0

## How It Works

For Beginners:
The Korteweg-de Vries equation is one of the most famous nonlinear PDEs in physics.
It describes waves in shallow water and is remarkable for having "soliton" solutions.

Variables:

- u(x,t) = Wave amplitude or displacement
- x = Spatial coordinate
- t = Time
- α = Nonlinear coefficient (strength of steepening)
- β = Dispersion coefficient (wave spreading)

Physical Interpretation:

- The u∂u/∂x term causes wave steepening (like shock waves)
- The ∂³u/∂x³ term causes dispersion (different frequencies travel at different speeds)
- When these effects balance, you get solitons - stable traveling wave packets

Solitons:

- Solitons maintain their shape while traveling at constant speed
- Two solitons can pass through each other without changing shape
- First observed by John Scott Russell in 1834 watching a wave in a canal

Standard Forms:

- Canonical form: ∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0 (α=6, β=1)
- Physical form: ∂u/∂t + u∂u/∂x + ∂³u/∂x³ = 0 (α=1, β=1)

Applications:

- Water waves in shallow channels
- Internal waves in oceans
- Plasma physics (ion-acoustic waves)
- Optical fiber communications
- Tsunami modeling (in simplified cases)

Example: A solitary wave traveling along a canal maintains its bell-shaped
profile indefinitely, unlike ordinary waves that disperse.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KortewegDeVriesEquation(,)` | Initializes a new instance of the Korteweg-de Vries Equation. |
| `KortewegDeVriesEquation(Double,Double)` | Initializes a new instance of the Korteweg-de Vries Equation with double parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDimension` |  |
| `Name` |  |
| `OutputDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Canonical` | Creates a Korteweg-de Vries Equation in canonical form (α=6, β=1). |
| `ComputeResidual(Vector<>,Vector<>,PDEDerivatives<>)` |  |
| `ComputeResidualGradient(Vector<>,Vector<>,PDEDerivatives<>)` |  |
| `Physical` | Creates a Korteweg-de Vries Equation in physical form (α=1, β=1). |

