---
title: "SincInterpolation<T>"
description: "Implements Sinc interpolation for 1D data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Sinc interpolation for 1D data points.

## For Beginners

Sinc interpolation is like creating a smooth curve through your data points
that preserves the frequency characteristics of your original data. It's particularly good
for signals like audio or sensor data where you want to maintain the original frequencies
when filling in gaps between known points. Think of it as drawing a curve that not only passes
through your points but also maintains the "rhythm" or "pattern" of your data.

## How It Works

Sinc interpolation is a technique based on the Whittaker–Shannon interpolation formula,
which is theoretically perfect for band-limited signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SincInterpolation(IEnumerable<Double>,IEnumerable<Double>,Double)` | Initializes a new instance of Sinc interpolation with the specified data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate()` | Interpolates a y-value for the given x coordinate using Sinc interpolation. |
| `Sinc()` | Calculates the Sinc function value: sin(px)/(px). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cutoffFrequency` | The cutoff frequency that controls the bandwidth of the interpolation. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-values at each data point. |

