---
title: "IPDESpecification<T>"
description: "Defines the interface for specifying Partial Differential Equations (PDEs) that can be used with Physics-Informed Neural Networks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Defines the interface for specifying Partial Differential Equations (PDEs) that can be used with Physics-Informed Neural Networks.

## How It Works

For Beginners:
A Partial Differential Equation (PDE) is an equation that involves rates of change with respect to multiple variables.
For example, the heat equation describes how temperature changes over both space and time.
This interface allows you to define any PDE in a way that neural networks can learn to solve it.

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDimension` | Gets the dimension of the input space (e.g., 2 for 2D spatial problems, 3 for 2D space + time). |
| `Name` | Gets the name or description of the PDE (e.g., "Heat Equation", "Navier-Stokes"). |
| `OutputDimension` | Gets the dimension of the output space (e.g., 1 for scalar fields like temperature, 3 for vector fields like velocity). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeResidual(Vector<>,Vector<>,PDEDerivatives<>)` | Computes the PDE residual at the given point. |

