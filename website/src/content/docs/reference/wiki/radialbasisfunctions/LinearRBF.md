---
title: "LinearRBF<T>"
description: "Implements a Linear Radial Basis Function (RBF) of the form f(r) = r."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Linear Radial Basis Function (RBF) of the form f(r) = r.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Linear RBF is the simplest possible RBF - it's just the distance itself. Think of it as a straight line
starting from zero at the center and growing uniformly as you move away in any direction.

This is different from most other RBFs because:

- Most RBFs have their highest value at the center and decrease with distance
- The Linear RBF starts at zero at the center and increases with distance
- Most RBFs have a "width" parameter to control how quickly they change with distance
- The Linear RBF has no width parameter - it always increases at the same rate

The Linear RBF is rarely used alone in practice, but it can be useful in certain specific applications
or as a building block for more complex functions.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that simply returns the radius itself.
Unlike most RBFs which reach their maximum at the center and decrease with distance, the Linear RBF
increases linearly with distance from the center. It has the simplest possible form of any RBF: f(r) = r.
Note that this function does not have a width parameter like most other RBFs.

The Linear RBF can be useful in specific applications where a direct proportionality to distance is desired.
It is also sometimes used as a component in more complex kernels or in combination with other RBFs.
However, it should be noted that this function does not approach zero as distance increases, which may
make it unsuitable for many typical RBF applications requiring localized influence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearRBF` | Initializes a new instance of the `LinearRBF` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Linear Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Linear RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Linear RBF with respect to a width parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

