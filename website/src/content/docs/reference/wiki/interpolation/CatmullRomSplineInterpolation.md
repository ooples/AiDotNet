---
title: "CatmullRomSplineInterpolation<T>"
description: "Implements Catmull-Rom spline interpolation for smooth curve fitting through a series of points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Catmull-Rom spline interpolation for smooth curve fitting through a series of points.

## How It Works

Catmull-Rom splines create smooth curves that pass through all the provided data points,
making them useful for animation paths, curve drawing, and data visualization.

**For Beginners:** Think of this as drawing a smooth curve through a set of dots. Unlike simpler
methods that might create sharp corners or jagged lines, Catmull-Rom splines create naturally
flowing curves that pass exactly through each point while maintaining smoothness.

Imagine connecting dots to draw the outline of a mountain range or a river - you want the
line to flow naturally between points rather than making sharp turns. That's what this
interpolation method provides.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CatmullRomSplineInterpolation(Vector<>,Vector<>,Double)` | Initializes a new instance of the CatmullRomSplineInterpolation class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CatmullRomSpline(,,,,)` | Calculates the Catmull-Rom spline value for a given parameter t and four control points. |
| `FindInterval()` | Finds the index of the interval containing the specified x-coordinate. |
| `Interpolate()` | Interpolates a y-value at the specified x-coordinate using Catmull-Rom spline interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Helper object for performing numeric operations on generic type T. |
| `_tension` | The tension parameter that controls the curvature of the spline. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |

