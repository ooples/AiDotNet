---
title: "KochanekBartelsSplineInterpolation<T>"
description: "Implements Kochanek-Bartels spline interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Kochanek-Bartels spline interpolation for one-dimensional data points.

## How It Works

Kochanek-Bartels splines (also known as TCB splines) provide control over the shape of the curve
through three parameters: tension, continuity, and bias. This allows for more flexible and
customizable interpolation compared to simpler methods.

**For Beginners:** This interpolation method creates smooth curves between data points with
special controls that let you adjust how the curve looks. Imagine drawing a line through
dots on a graph, but being able to control how "tight" the curve is, how smoothly it
transitions between segments, and whether it tends to overshoot or undershoot. This is
particularly useful for animation paths or when you need precise control over the shape
of a curve.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KochanekBartelsSplineInterpolation(Vector<>,Vector<>,Double,Double,Double)` | Creates a new Kochanek-Bartels spline interpolation from the given data points and parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateTangent(,,)` | Calculates the tangent at a point for the Kochanek-Bartels spline. |
| `FindInterval()` | Finds the index of the interval containing the given x-value. |
| `Interpolate()` | Calculates the interpolated y-value for a given x-value using Kochanek-Bartels spline interpolation. |
| `KochanekBartelsSpline(,,,,)` | Calculates the value of the Kochanek-Bartels spline at a point within a segment. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | Controls whether the curve tends to overshoot (negative values) or undershoot (positive values). |
| `_continuity` | Controls the smoothness of transitions between curve segments. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_tension` | Controls how "tight" the curve is. |
| `_x` | The x-coordinates of the data points (independent variable). |
| `_y` | The y-coordinates of the data points (dependent variable). |

