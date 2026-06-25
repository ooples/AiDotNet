---
title: "CubicConvolutionInterpolation<T>"
description: "Implements cubic convolution interpolation for 2D data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements cubic convolution interpolation for 2D data points.

## How It Works

Cubic convolution interpolation creates smooth surfaces from a grid of data points.
It uses 16 neighboring points (a 4x4 grid) to calculate each interpolated value,
resulting in a continuous surface with smooth first derivatives.

**For Beginners:** This class helps you estimate values between known data points on a 2D grid.
Imagine having temperature readings at specific locations on a map, and you want to
estimate the temperature at locations between your measurement points. This interpolation
creates a smooth surface that passes through all your known points and provides reasonable
estimates for the points in between.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CubicConvolutionInterpolation(Vector<>,Vector<>,Matrix<>)` | Creates a new cubic convolution interpolation from the given 2D data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BicubicInterpolate([0:,0:],,)` | Performs bicubic interpolation using a 4x4 grid of points. |
| `CubicInterpolate(,,,,)` | Performs cubic interpolation between four points. |
| `FindInterval(Vector<>,)` | Finds the index of the interval in a sorted array that contains the given point. |
| `Interpolate(,)` | Calculates the interpolated z-value for a given (x,y) point using cubic convolution. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |
| `_z` | The z-values (heights) at each (x,y) grid point. |

