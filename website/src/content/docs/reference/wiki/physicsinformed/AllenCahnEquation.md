---
title: "AllenCahnEquation<T>"
description: "Represents the Allen-Cahn equation: u_t - epsilon^2 * u_xx + u^3 - u = 0."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Allen-Cahn equation: u_t - epsilon^2 * u_xx + u^3 - u = 0.

## How It Works

For Beginners:
The Allen-Cahn equation models phase separation and interface motion.
It combines diffusion (smoothing) with a nonlinear reaction term.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AllenCahnEquation()` | Initializes a new instance of the Allen-Cahn equation. |
| `AllenCahnEquation(Double)` | Initializes a new instance of the Allen-Cahn equation with double parameter. |

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

