---
title: "SphericalRBF<T>"
description: "Implements a Spherical Radial Basis Function (RBF) with compact support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Spherical Radial Basis Function (RBF) with compact support.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Spherical RBF has a unique property compared to many other RBFs: it becomes exactly zero
beyond a certain distance (the support radius). Think of it like a hill that completely flattens
out beyond a specific distance - there's a clear boundary where the function's influence stops.

Inside its support radius, the function has a curved shape that smoothly transitions to zero at the boundary:

- At the center (distance = 0), the value is exactly 1
- As distance increases, the value decreases in a curved pattern
- At exactly the support radius (epsilon), the value becomes 0
- Beyond the support radius, the value stays at 0

This "limited reach" property makes the Spherical RBF computationally efficient for large datasets,
as points beyond the support radius can be completely ignored in calculations.

## How It Works

This class provides an implementation of a Spherical Radial Basis Function, which is a compactly
supported RBF defined as:
f(r) = 1 - 1.5(r/e) + 0.5(r/e)³ for r = e
f(r) = 0 for r > e
where r is the radial distance and e (epsilon) is a shape parameter controlling the support radius.

Unlike many other RBFs that have non-zero values for all distances, the Spherical RBF becomes exactly
zero beyond a certain radius (e), giving it "compact support." This property can be computationally
advantageous when working with large datasets, as it leads to sparse matrices in many applications.
The function is C² continuous, meaning it has continuous derivatives up to order 2.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SphericalRBF(Double)` | Initializes a new instance of the `SphericalRBF` class with a specified support radius. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Spherical Radial Basis Function for a given radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Spherical RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the support radius of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

