---
title: "PDESpecificationBase<T>"
description: "Base class for all Partial Differential Equation (PDE) specifications."
section: "API Reference"
---

`Base Classes` · `AiDotNet.PhysicsInformed.PDEs`

Base class for all Partial Differential Equation (PDE) specifications.
Provides common functionality for PDE implementations used with Physics-Informed Neural Networks.

## For Beginners

A Partial Differential Equation (PDE) describes how a quantity changes
with respect to multiple variables (like space and time). This base class provides the
foundation for implementing any PDE that can be solved using Physics-Informed Neural Networks.

## How It Works

All PDE implementations should inherit from this class rather than implementing
`IPDESpecification` directly. This ensures consistent behavior and
provides access to common helper methods.

To create a new PDE, override the abstract members:

- `PDEDerivatives{` - Calculate how much the PDE is violated
- `InputDimension` - Number of independent variables (space + time)
- `OutputDimension` - Number of dependent variables (solution components)
- `Name` - Human-readable name of the PDE

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PDESpecificationBase` | Initializes a new instance of the `PDESpecificationBase` class. |

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
| `ComputeTapeResidual(Tensor<>,Tensor<>,IEngine)` | Computes the PDE residual using tape-differentiable engine operations. |
| `CreateGradient` | Creates a new PDEResidualGradient with the appropriate dimensions. |
| `ValidateFirstDerivatives(PDEDerivatives<>)` | Validates that first-order derivatives are available. |
| `ValidateNonNegative(,String)` | Validates that a parameter is non-negative (greater than or equal to zero). |
| `ValidatePositive(,String)` | Validates that a parameter is positive (greater than zero). |
| `ValidateSecondDerivatives(PDEDerivatives<>)` | Validates that both first and second-order derivatives are available. |
| `ValidateThirdDerivatives(PDEDerivatives<>)` | Validates that first, second, and third-order derivatives are available. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

