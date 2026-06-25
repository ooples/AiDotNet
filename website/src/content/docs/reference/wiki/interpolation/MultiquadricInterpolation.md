---
title: "MultiquadricInterpolation<T>"
description: "Implements Multiquadric Radial Basis Function interpolation for two-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Multiquadric Radial Basis Function interpolation for two-dimensional data points.

## How It Works

Multiquadric interpolation is a powerful technique for creating smooth surfaces from scattered data points.
It uses radial basis functions centered at each data point to construct an interpolating surface.

**For Beginners:** Imagine you have several points with known heights (like mountains on a map),
and you want to estimate the height at any location between these points. Multiquadric interpolation
creates a smooth surface that passes exactly through all your known points while providing reasonable
estimates for all the areas in between. It's particularly good at handling irregularly spaced data points
and can create very smooth surfaces.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiquadricInterpolation(Vector<>,Vector<>,Vector<>,Double,IMatrixDecomposition<>)` | Creates a new instance of the Multiquadric interpolation algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCoefficients` | Calculates the coefficients needed for the multiquadric interpolation. |
| `Interpolate(,)` | Interpolates the z-value at a given (x,y) coordinate using Multiquadric interpolation. |
| `MultiquadricBasis()` | Calculates the multiquadric radial basis function for a given distance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The calculated coefficients for the radial basis functions. |
| `_decomposition` | Optional matrix decomposition method for solving the interpolation system. |
| `_epsilon` | The shape parameter that controls the smoothness of the interpolation. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates of the known data points. |
| `_z` | The z-values (heights) of the known data points. |

