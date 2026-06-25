---
title: "HermiteInterpolation<T>"
description: "Implements Hermite interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Hermite interpolation for one-dimensional data points.

## How It Works

Hermite interpolation creates a smooth curve that passes through all given data points
while also matching specified derivatives (slopes) at those points. This provides more
control over the shape of the curve compared to simpler interpolation methods.

**For Beginners:** This class helps you estimate values between known data points when you
know not only the values at certain points but also how quickly those values are changing
(the slopes) at those points. Imagine drawing a smooth curve through dots on a graph,
but also controlling which direction the curve is heading as it passes through each dot.
This gives you a more natural-looking curve that respects both the values and their rates
of change.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HermiteInterpolation(Vector<>,Vector<>,Vector<>)` | Creates a new Hermite interpolation from the given data points and slopes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindInterval()` | Finds the index of the interval containing the given x-value. |
| `Interpolate()` | Calculates the interpolated y-value for a given x-value using Hermite interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_m` | The slopes (derivatives) at each data point. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points (independent variable). |
| `_y` | The y-coordinates of the data points (dependent variable). |

