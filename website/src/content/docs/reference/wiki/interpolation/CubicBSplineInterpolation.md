---
title: "CubicBSplineInterpolation<T>"
description: "Implements cubic B-spline interpolation for smooth curve fitting through data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements cubic B-spline interpolation for smooth curve fitting through data points.

## How It Works

Cubic B-spline interpolation creates a smooth curve that passes through or near all provided data points.
It's particularly useful for creating natural-looking curves with continuous first and second derivatives.

**For Beginners:** B-splines are a special type of smooth curve used in computer graphics and data analysis.
Unlike simpler interpolation methods, B-splines create exceptionally smooth curves that don't have
sudden changes in direction. Think of them as drawing a smooth line through your points using a
flexible ruler that naturally creates gentle curves.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CubicBSplineInterpolation(Vector<>,Vector<>,Int32,MatrixDecompositionType)` | Creates a new cubic B-spline interpolation from the given data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBasisFunctions(,Int32)` | Calculates the basis functions for a B-spline at a specific point. |
| `CalculateCoefficients` | Calculates the coefficients that define the B-spline curve. |
| `FindSpan()` | Finds the knot span index that contains the given x-value. |
| `GenerateKnots` | Generates the knot vector for the B-spline curve. |
| `Interpolate()` | Calculates the interpolated y-value for a given x-value using the B-spline curve. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The calculated coefficients that define the B-spline curve. |
| `_decompositionType` | The type of matrix decomposition used for solving the linear system. |
| `_degree` | The degree of the B-spline curve (default is 3 for cubic). |
| `_knots` | The knot vector that defines the B-spline curve. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |

