---
title: "MovingLeastSquaresInterpolation<T>"
description: "Implements Moving Least Squares interpolation for two-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Moving Least Squares interpolation for two-dimensional data points.

## How It Works

Moving Least Squares (MLS) is a method for smoothly interpolating scattered data points
in two dimensions. It creates a continuous surface that passes through or near the original data points.

**For Beginners:** Imagine you have several points with known heights (like mountains on a map),
and you want to estimate the height at any location between these points. Moving Least Squares
creates a smooth surface that respects your known points while providing reasonable estimates
for all the areas in between. It's like creating a smooth terrain from a set of elevation measurements.

Unlike simpler methods, MLS adapts to the local density and arrangement of your data points,
giving more weight to nearby points when estimating a value at a specific location.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MovingLeastSquaresInterpolation(Vector<>,Vector<>,Vector<>,Double,Int32,IMatrixDecomposition<>)` | Creates a new instance of the Moving Least Squares interpolation algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateWeight()` | Calculates the weight for a data point based on its distance from the target point. |
| `Interpolate(,)` | Interpolates the z-value at a given (x,y) coordinate using Moving Least Squares. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decomposition` | Optional matrix decomposition method for solving the least squares system. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_polynomialDegree` | The degree of the polynomial used for local approximation. |
| `_smoothingLength` | Controls how far the influence of each data point extends. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates of the known data points. |
| `_z` | The z-values (heights) of the known data points. |

