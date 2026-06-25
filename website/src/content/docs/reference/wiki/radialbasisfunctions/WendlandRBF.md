---
title: "WendlandRBF<T>"
description: "Implements Wendland's compactly supported Radial Basis Functions with different smoothness orders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements Wendland's compactly supported Radial Basis Functions with different smoothness orders.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

Wendland RBFs are a family of functions that have two important properties:

1. They have "compact support" - they become exactly zero beyond a certain distance (the support radius)
2. They are very smooth - they transition gradually to zero at the boundary with no sudden changes

Think of them like smooth hills or bumps that are exactly flat (zero) beyond a certain distance.
You can choose from different types of Wendland functions based on how smooth you need them to be:

- k = 0: The basic version, reasonably smooth but with limited continuity
- k = 1: A smoother version, with more continuous derivatives
- k = 2: The smoothest version, with even more continuous derivatives

The higher the k value, the smoother the function, but also the more computationally expensive.
These functions are particularly useful in scientific computing because they combine efficiency
(from the compact support) with high quality results (from the smoothness).

## How It Works

This class provides an implementation of Wendland's family of compactly supported Radial Basis Functions.
These functions are defined by a smoothness parameter k and have the form:

- For k = 0: f(r) = (1 - r)² (for r = 1, 0 otherwise)
- For k = 1: f(r) = (1 - r)4(1 + 4r) (for r = 1, 0 otherwise)
- For k = 2: f(r) = (1 - r)6(3 + 18r + 35r²) (for r = 1, 0 otherwise)

where r is the normalized radial distance (actual distance divided by the support radius).

Wendland functions are popular in scientific computing because they combine compact support
(they become exactly zero beyond a certain radius) with high order smoothness properties. The parameter k
controls the smoothness of the function: higher k values yield more derivatives at r = 0 and r = 1,
resulting in smoother interpolations. These functions are particularly useful for scattered data
interpolation, meshless methods for solving PDEs, and computer graphics applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WendlandRBF(Int32,Double)` | Initializes a new instance of the `WendlandRBF` class with specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Wendland Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Wendland RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Wendland RBF with respect to the support radius parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_k` | The smoothness parameter controlling the order of the Wendland function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |
| `_supportRadius` | The support radius beyond which the function becomes zero. |

