---
title: "RadialBasisFunctionInterpolation<T>"
description: "Implements Radial Basis Function (RBF) interpolation for 2D data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Radial Basis Function (RBF) interpolation for 2D data points.

## For Beginners

Think of RBF interpolation as creating a rubber sheet that's stretched to
pass through all your data points. The sheet's shape between and beyond your known points
is determined by special mathematical functions (called radial basis functions) that create
smooth transitions. This is particularly useful when your data points aren't arranged in a grid.

## How It Works

Radial Basis Function interpolation is a powerful technique for creating smooth surfaces
that pass through scattered data points in 2D or higher dimensions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RadialBasisFunctionInterpolation(Vector<>,Vector<>,Vector<>,IRadialBasisFunction<>,IMatrixDecomposition<>)` | Initializes a new instance of the RBF interpolation with the specified data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDistance(,,,)` | Calculates the Euclidean distance between two points in 2D space. |
| `CalculateWeights` | Calculates the weights for each radial basis function. |
| `Interpolate(,)` | Interpolates a z-value for the given x and y coordinates using RBF interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decomposition` | The matrix decomposition method used to solve the linear system. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_rbf` | The radial basis function used for interpolation. |
| `_weights` | The calculated weights for each radial basis function. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |
| `_z` | The z-values (heights) at each data point. |

