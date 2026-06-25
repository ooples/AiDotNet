---
title: "PchipInterpolation<T>"
description: "Implements Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.

## For Beginners

PCHIP is a method that creates a smooth curve through your data points.
Unlike some other methods, it avoids creating artificial "wiggles" in the curve,
making it particularly useful for scientific data where you want to maintain the
general shape and trends of your original data.

## How It Works

PCHIP interpolation creates a smooth curve that passes through all data points while
preserving the shape of the data, particularly maintaining monotonicity (keeping the same
direction of increase or decrease between points).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PchipInterpolation(Vector<>,Vector<>)` | Initializes a new instance of the PCHIP interpolation with the specified data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSlopes` | Calculates the slopes at each data point to ensure a smooth, shape-preserving curve. |
| `FindInterval()` | Finds the appropriate interval in the data points for the given x-value. |
| `Interpolate()` | Interpolates a y-value for the given x-value using PCHIP interpolation. |
| `WeightedHarmonicMean(,)` | Calculates a weighted harmonic mean of two values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_slopes` | The calculated slopes at each data point. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |

