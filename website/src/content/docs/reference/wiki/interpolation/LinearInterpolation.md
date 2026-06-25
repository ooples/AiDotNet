---
title: "LinearInterpolation<T>"
description: "Implements linear interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements linear interpolation for one-dimensional data points.

## How It Works

Linear interpolation is the simplest form of interpolation, connecting data points with straight lines.
It estimates values between known data points by assuming a straight line between them.

**For Beginners:** Linear interpolation is like drawing straight lines between dots on a graph.
If you have two points (like one at x=1, y=10 and another at x=3, y=20), and you want to know
what the y-value would be at x=2, linear interpolation would give you y=15 because it's exactly
halfway between the two known points. It's the simplest and most intuitive way to estimate values
between known points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearInterpolation(Vector<>,Vector<>)` | Creates a new instance of the linear interpolation algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindInterval()` | Finds the interval in the x-coordinates array that contains the given x-value. |
| `Interpolate()` | Interpolates the y-value at a given x-coordinate using linear interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates (values) of the known data points. |

