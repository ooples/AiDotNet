---
title: "ThinPlateSplineInterpolation<T>"
description: "Implements Thin Plate Spline interpolation for 2D scattered data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Thin Plate Spline interpolation for 2D scattered data points.

## For Beginners

Think of Thin Plate Spline interpolation like placing a flexible sheet of metal
over a set of pins (your data points) at different heights. The metal bends to touch all pins
while maintaining the smoothest possible surface between them. This method is excellent for
creating smooth surfaces from scattered measurements, such as elevation data or temperature
readings taken at irregular locations.

## How It Works

Thin Plate Spline (TPS) is a technique for interpolating smooth surfaces through scattered data points
in two dimensions. It minimizes the "bending energy" of the surface, creating a result that is
analogous to the shape a thin metal plate would take if forced to pass through all the data points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThinPlateSplineInterpolation(Vector<>,Vector<>,Vector<>,IMatrixDecomposition<>)` | Initializes a new instance of Thin Plate Spline interpolation with the specified data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDistance(,,,)` | Calculates the Euclidean distance between two points in 2D space. |
| `CalculateWeights` | Calculates the weights and coefficients needed for the interpolation. |
| `Interpolate(,)` | Interpolates a z-value for the given x and y coordinates using Thin Plate Spline interpolation. |
| `RadialBasisFunction()` | Calculates the radial basis function used in Thin Plate Spline interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_a0` | The constant term in the polynomial part of the TPS equation. |
| `_ax` | The coefficient for x in the polynomial part of the TPS equation. |
| `_ay` | The coefficient for y in the polynomial part of the TPS equation. |
| `_decomposition` | The matrix decomposition method used to solve the linear system. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_weights` | The weights calculated for each data point used in the interpolation. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |
| `_z` | The z-values (heights) at each data point. |

