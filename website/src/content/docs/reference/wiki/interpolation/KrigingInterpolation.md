---
title: "KrigingInterpolation<T>"
description: "Implements Kriging interpolation for two-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Kriging interpolation for two-dimensional data points.

## How It Works

Kriging is a geostatistical interpolation technique that predicts unknown values
based on the spatial correlation between known data points. It's particularly useful
for creating smooth surfaces from scattered data points.

**For Beginners:** Kriging is like predicting the height of a landscape at any point
when you only know the heights at certain locations. It works by assuming that points
closer together are more likely to have similar values than points far apart.
This method is widely used in geography, mining, and environmental science.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KrigingInterpolation(Vector<>,Vector<>,Vector<>,IKernelFunction<>,IMatrixDecomposition<>)` | Creates a new instance of the Kriging interpolation algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDistance(,,,)` | Calculates the Euclidean distance between two points in 2D space. |
| `CalculateWeights` | Calculates the weights used in the Kriging interpolation. |
| `EstimateVariogramParameters` | Estimates the parameters of the variogram model from the data. |
| `FitExponentialVariogram(List<>,List<>)` | Fits an exponential variogram model to the provided distance and gamma value pairs. |
| `Interpolate(,)` | Interpolates the z-value at a given (x,y) coordinate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decomposition` | The matrix decomposition method used for solving linear systems. |
| `_kernel` | The kernel function that determines how the influence of points decreases with distance. |
| `_nugget` | The nugget parameter of the variogram model, representing measurement error or micro-scale variation. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_range` | The range parameter of the variogram model, representing the distance at which points become uncorrelated. |
| `_sill` | The sill parameter of the variogram model, representing the maximum variance between points. |
| `_weights` | The calculated weights used in the Kriging interpolation. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates of the known data points. |
| `_z` | The z-values (heights) of the known data points. |

