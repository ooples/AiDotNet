---
title: "MonotoneCubicInterpolation<T>"
description: "Implements monotone cubic interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements monotone cubic interpolation for one-dimensional data points.

## How It Works

Monotone cubic interpolation creates a smooth curve through data points while preserving
the monotonicity of the data (meaning if your data is increasing, the interpolation will also
be increasing, and similarly for decreasing data).

**For Beginners:** Monotone cubic interpolation is like drawing a smooth curve through points
on a graph, but with a special property: if your original data is always going up (or always going down)
between certain points, the curve will also always go up (or down) between those points. This avoids
unwanted "wiggles" or oscillations that can happen with other smooth interpolation methods.
It's particularly useful when you know your data should never "change direction" between points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MonotoneCubicInterpolation(Vector<>,Vector<>)` | Creates a new instance of the monotone cubic interpolation algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSlopes` | Calculates the slopes at each data point to ensure monotonicity. |
| `FindInterval()` | Finds the interval in the x-coordinates array that contains the given x-value. |
| `Interpolate()` | Interpolates the y-value at a given x-coordinate using monotone cubic interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_m` | The slopes at each data point, calculated to ensure monotonicity. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates (values) of the known data points. |

