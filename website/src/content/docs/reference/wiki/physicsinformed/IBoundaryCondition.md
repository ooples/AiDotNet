---
title: "IBoundaryCondition<T>"
description: "Defines boundary conditions for a PDE problem."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Defines boundary conditions for a PDE problem.

## How It Works

For Beginners:
Boundary conditions specify what happens at the edges of your problem domain.
For example, in a heat equation, you might specify the temperature at the boundaries of a rod.
Common types:

- Dirichlet: The value is specified at the boundary (e.g., temperature = 100°C)
- Neumann: The derivative is specified at the boundary (e.g., heat flux = 0, meaning insulated)
- Robin: A combination of value and derivative

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the boundary (e.g., "Left Wall", "Top Edge"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBoundaryResidual(Vector<>,Vector<>,PDEDerivatives<>)` | Computes the boundary condition residual. |
| `IsOnBoundary(Vector<>)` | Determines if a point is on the boundary. |

