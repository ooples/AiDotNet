---
title: "AdaptiveCubicSplineInterpolation<T>"
description: "Provides an adaptive cubic spline interpolation that automatically switches between natural and monotonic  cubic splines based on data characteristics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Provides an adaptive cubic spline interpolation that automatically switches between natural and monotonic 
cubic splines based on data characteristics.

## How It Works

This class implements an intelligent interpolation strategy that combines the smoothness of natural 
cubic splines with the shape-preserving properties of monotonic cubic splines.

**For Beginners:** Interpolation is like "connecting the dots" between data points to create a smooth curve.

Imagine you have several points on a graph and want to draw a smooth line through them:

- Natural cubic splines create very smooth curves but might overshoot or create waves where your data doesn't have them
- Monotonic splines preserve the "shape" of your data (keeping increasing parts increasing, etc.) but might be less smooth

This adaptive method gives you the best of both worlds by:

1. Looking at each segment of your data
2. Deciding whether to use the smoother method or the shape-preserving method for that segment
3. Automatically switching between methods as needed

It's like having a smart drawing assistant that chooses the right tool for each part of your curve!

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveCubicSplineInterpolation(Vector<>,Vector<>,)` | Initializes a new instance of the AdaptiveCubicSplineInterpolation class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetermineInterpolationMethod()` | Determines which interpolation method to use for each interval based on the threshold. |
| `FindInterval()` | Finds the interval index that contains the specified x-coordinate. |
| `Interpolate()` | Interpolates a value at the specified x-coordinate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_monotonicSpline` | The monotonic cubic spline interpolation instance. |
| `_naturalSpline` | The natural cubic spline interpolation instance. |
| `_numOps` | Helper object for performing numeric operations on generic type T. |
| `_useMonotonic` | Array indicating which interpolation method to use for each interval. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |

