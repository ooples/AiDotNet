---
title: "IMultiScalePDE<T>"
description: "Defines the interface for multi-scale Partial Differential Equations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Defines the interface for multi-scale Partial Differential Equations.

## How It Works

For Beginners:
Multi-scale PDEs describe phenomena that occur across multiple length or time scales.
Examples include:

- Turbulent flows (large eddies to small vortices)
- Materials science (atomic to macroscopic behavior)
- Climate modeling (local weather to global patterns)
- Biological systems (molecular to tissue level)

Why Multi-scale is Challenging:
Traditional single-scale methods struggle because:

1. Fine scale requires tiny mesh elements (expensive)
2. Coarse scale misses important details
3. Different scales have different time dynamics

Multi-scale Solution Strategy:

1. Decompose solution into scale components: u = u_coarse + u_fine
2. Learn each scale with appropriate resolution
3. Couple scales through cross-scale interactions
4. Use appropriate loss weights for each scale

Key Concepts:

- Characteristic Length Scale: The typical size of features at each scale
- Scale Separation: When scales are well-separated (e.g., 10x ratio)
- Cross-scale Coupling: How different scales interact
- Homogenization: Averaging fine-scale behavior for coarse-scale equations

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfScales` | Gets the number of scales in the problem. |
| `ScaleCharacteristicLengths` | Gets the characteristic length scales of the problem. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScaleCoupling(Int32,Int32,[],[],[],PDEDerivatives<>,PDEDerivatives<>)` | Computes the coupling term between two scales. |
| `ComputeScaleResidual(Int32,[],[],PDEDerivatives<>)` | Computes the PDE residual at a specific scale. |
| `GetScaleLossWeight(Int32)` | Gets the recommended loss weight for each scale. |
| `GetScaleOutputDimension(Int32)` | Gets the output dimension for a specific scale. |

