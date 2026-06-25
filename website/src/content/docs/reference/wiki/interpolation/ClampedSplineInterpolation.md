---
title: "ClampedSplineInterpolation<T>"
description: "Implements clamped cubic spline interpolation for smooth curve fitting with controlled endpoints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements clamped cubic spline interpolation for smooth curve fitting with controlled endpoints.

## How It Works

Clamped cubic splines create smooth curves that pass through all provided data points
while allowing you to specify the slope at the endpoints of the curve.

**For Beginners:** Think of this as drawing a smooth curve through a set of points where
you can control the "direction" the curve enters and exits the first and last points.
This is useful when you need the curve to approach the endpoints from specific angles,
such as when connecting to other curves or when modeling physical phenomena with known
behavior at the boundaries.

Unlike other interpolation methods, clamped splines give you control over how the curve
behaves at its edges, making them ideal for scenarios where the slope at the endpoints
is known or important.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClampedSplineInterpolation(Vector<>,Vector<>,Double,Double,IMatrixDecomposition<>)` | Initializes a new instance of the ClampedSplineInterpolation class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCoefficients` | Calculates the coefficients needed for the cubic spline interpolation. |
| `FindInterval()` | Finds the interval in the data that contains the given x-coordinate. |
| `Interpolate()` | Interpolates a y-value at the specified x-coordinate using clamped cubic spline interpolation. |
| `Power(,Int32)` | Raises a value to the specified power through repeated multiplication. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The coefficients of the cubic polynomials that define the spline segments. |
| `_decomposition` | The matrix decomposition method used to solve the spline equations. |
| `_endSlope` | The slope of the curve at the last data point. |
| `_numOps` | Helper object for performing numeric operations on generic type T. |
| `_startSlope` | The slope of the curve at the first data point. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |

