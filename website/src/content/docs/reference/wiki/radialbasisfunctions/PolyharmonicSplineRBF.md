---
title: "PolyharmonicSplineRBF<T>"
description: "Implements a Polyharmonic Spline Radial Basis Function (RBF) with different forms based on a parameter k."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Polyharmonic Spline Radial Basis Function (RBF) with different forms based on a parameter k.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Polyharmonic Spline RBF is used to create smooth curves or surfaces that pass through a set of points.
Think of it like drawing a smooth line through dots on a page, but it works in any number of dimensions.

This particular RBF comes in different "flavors" depending on the value of parameter k:

- When k is odd (1, 3, 5, etc.): The function is simply the distance raised to the power of k
- When k is even (2, 4, 6, etc.): The function is the distance raised to the power of k, multiplied by the logarithm of the distance

A unique property of polyharmonic splines is that they don't have a width parameter like most other RBFs.
This makes them "scale-invariant" - scaling all your input distances by the same factor only changes
the output by a constant factor, which doesn't affect the shape of the resulting interpolation.

Common choices for k include:

- k = 1: "Linear" (r)
- k = 2: "Thin plate spline" (r² log r)
- k = 3: "Cubic" (r³)

## How It Works

This class provides an implementation of Polyharmonic Spline Radial Basis Functions, which are defined as:
f(r) = r^k for odd k
f(r) = r^k * log(r) for even k
where r is the radial distance and k is an integer parameter (typically k = 1).

Polyharmonic splines are used in scattered data interpolation, numerical solutions of partial differential
equations, and image processing. They are particularly useful for problems in multiple dimensions
due to their theoretical properties. Unlike many other RBFs, polyharmonic splines do not have a width
parameter, making them scale-invariant. The parameter k controls the smoothness of the resulting
interpolation, with higher values of k producing smoother functions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolyharmonicSplineRBF(Int32)` | Initializes a new instance of the `PolyharmonicSplineRBF` class with a specified k parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Polyharmonic Spline Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Polyharmonic Spline RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Polyharmonic Spline RBF with respect to a width parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_k` | The parameter k that determines the type and order of the polyharmonic spline. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

