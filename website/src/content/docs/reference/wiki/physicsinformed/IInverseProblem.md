---
title: "IInverseProblem<T>"
description: "Defines the interface for inverse problems in physics-informed neural networks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Defines the interface for inverse problems in physics-informed neural networks.

## How It Works

For Beginners:
An inverse problem is about finding unknown causes from observed effects.

Forward Problem (typical):

- Known: Initial conditions, boundary conditions, physical parameters
- Find: Solution at all points in space and time
- Example: Given thermal conductivity k, find temperature distribution T(x,t)

Inverse Problem:

- Known: Some observations of the solution
- Find: Unknown physical parameters or hidden fields
- Example: Given temperature measurements, find thermal conductivity k

Types of Inverse Problems:

1. Parameter Identification:
- Find unknown constants in the PDE
- Example: Identify diffusion coefficient from concentration data

2. Source Identification:
- Find unknown source terms
- Example: Locate pollution source from downstream measurements

3. Boundary Identification:
- Determine unknown boundary conditions
- Example: Infer surface heat flux from internal temperature sensors

4. Geometry Identification:
- Find unknown shape of domain
- Example: Detect tumor location from external measurements

Challenges:

1. Ill-posedness: Small noise in data → large errors in parameters
2. Non-uniqueness: Multiple parameter values may fit the data
3. Regularization: Need to impose constraints for stable solutions

PINN Advantage for Inverse Problems:

- Learns solution AND parameters simultaneously
- Physics constraints act as regularization
- Can handle noisy and sparse data
- No need for iterative PDE solves

## Properties

| Property | Summary |
|:-----|:--------|
| `HasMeasurementNoiseLevel` | Gets whether the measurement noise level is known. |
| `InitialParameterGuesses` | Gets initial guesses for the unknown parameters. |
| `MeasurementNoiseLevel` | Gets the measurement noise level (if known). |
| `NumberOfParameters` | Gets the number of unknown parameters. |
| `Observations` | Gets the observation data points. |
| `ParameterLowerBounds` | Gets lower bounds for the parameters (for constrained optimization). |
| `ParameterNames` | Gets the names of the unknown parameters to identify. |
| `ParameterUpperBounds` | Gets upper bounds for the parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateParameterizedPDE([])` | Applies parameters to the underlying PDE and returns the modified PDE specification. |
| `ValidateParameters([])` | Validates that the parameter values are physically meaningful. |

