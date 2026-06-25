---
title: "BilinearInterpolation<T>"
description: "Implements Bilinear Interpolation for estimating values between points in a 2D grid."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Bilinear Interpolation for estimating values between points in a 2D grid.

## How It Works

Bilinear interpolation creates a smooth surface that passes through a grid of known data points.
It's commonly used for image resizing, terrain modeling, and data visualization.

**For Beginners:** Think of bilinear interpolation as a way to "fill in the blanks" between points in a grid.
Imagine you have a grid of temperature readings taken at different locations, but you want to know
the temperature at a location between your measurement points. Bilinear interpolation gives you
an estimated value based on the four nearest known points.

Unlike simply picking the closest point's value, bilinear interpolation creates a smooth blend
between all four surrounding points, giving a more natural and accurate estimate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BilinearInterpolation(Vector<>,Vector<>,Matrix<>)` | Initializes a new instance of the BilinearInterpolation class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindInterval(Vector<>,)` | Finds the index of the interval containing the specified point. |
| `Interpolate(,)` | Interpolates a value at the specified (x,y) coordinates. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Helper object for performing numeric operations on generic type T. |
| `_x` | The x-coordinates of the grid points. |
| `_y` | The y-coordinates of the grid points. |
| `_z` | The z-values (data values) at each grid point, organized as a matrix. |

