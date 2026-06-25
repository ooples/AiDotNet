---
title: "LanczosInterpolation<T>"
description: "Implements Lanczos interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Lanczos interpolation for one-dimensional data points.

## How It Works

Lanczos interpolation is a high-quality resampling technique that uses a windowed sinc function
to create smooth interpolations between data points. It's commonly used in image and signal processing.

**For Beginners:** Lanczos interpolation is like a sophisticated way of estimating values between known points.
Imagine you have several dots on a graph and want to draw a smooth curve through them. Lanczos uses a special
mathematical approach that creates a natural-looking curve while preserving important details in your data.
It's particularly good at maintaining sharp edges while still creating smooth transitions, which is why
it's popular for resizing images and processing signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LanczosInterpolation(Vector<>,Vector<>,Int32)` | Creates a new instance of the Lanczos interpolation algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate()` | Interpolates the y-value at a given x-coordinate using Lanczos interpolation. |
| `LanczosKernel()` | Calculates the Lanczos kernel value for a given distance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_a` | The 'a' parameter that controls the size of the Lanczos window. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates (values) of the known data points. |

