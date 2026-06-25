---
title: "BicubicInterpolation<T>"
description: "Implements Bicubic Interpolation for estimating values between points in a 2D grid."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Bicubic Interpolation for estimating values between points in a 2D grid.

## How It Works

Bicubic interpolation creates a smooth surface that passes through a grid of known data points.
It's commonly used for image resizing, terrain modeling, and scientific data visualization.

**For Beginners:** Think of this as a sophisticated way to "fill in the blanks" between points in a grid.
Imagine you have a grid of height measurements for a landscape (like a topographic map) but you want
to know the height at points between your measurements. Bicubic interpolation creates a smooth surface
that passes through all your known points and gives reasonable estimates for the in-between areas.

Unlike simpler methods, bicubic interpolation considers not just the nearest points but also how the
surface is changing (its "slope" and "curvature"), resulting in smoother, more natural-looking results.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BicubicInterpolation(Vector<>,Vector<>,Matrix<>,IMatrixDecomposition<>)` | Initializes a new instance of the BicubicInterpolation class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BinarySearchExact(Vector<>,)` | Binary search for an exact match in a sorted array. |
| `CalculateBicubicCoefficients([0:,0:])` | Calculates the bicubic coefficients from a 4x4 patch of points. |
| `FindInterval(Vector<>,)` | Finds the index of the interval containing the specified point. |
| `Interpolate(,)` | Interpolates a value at the specified (x,y) coordinates. |
| `InterpolateBicubicPatch([0:,0:],,)` | Performs the actual bicubic interpolation on a 4x4 patch of points. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decomposition` | Optional matrix decomposition method used for solving the linear system. |
| `_numOps` | Helper object for performing numeric operations on generic type T. |
| `_x` | The x-coordinates of the grid points. |
| `_y` | The y-coordinates of the grid points. |
| `_z` | The z-values (heights) at each grid point, organized as a matrix. |

