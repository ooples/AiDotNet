---
title: "I2DInterpolation<T>"
description: "Defines an interface for two-dimensional interpolation algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for two-dimensional interpolation algorithms.

## How It Works

**For Beginners:** Interpolation is like "filling in the blanks" between known data points.

Imagine you have a table of temperatures measured at specific locations on a map (x,y coordinates).
But what if you want to know the temperature at a location where you don't have a measurement?

2D interpolation helps you estimate that value based on the surrounding known values.
It's similar to how weather maps show smooth color gradients between weather stations.

This interface defines a standard way to perform this estimation for any type of
two-dimensional interpolation algorithm.

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(,)` | Calculates an interpolated value at the specified coordinates. |

