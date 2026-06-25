---
title: "BSplineKernel<T>"
description: "Implements the B-Spline kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the B-Spline kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The B-Spline kernel is a specialized similarity measure that works well for data that needs smooth
transitions between points. Think of B-splines like drawing a smooth curve through a set of points,
where the curve doesn't necessarily pass through all points but creates a smooth approximation.

## How It Works

The B-Spline kernel is based on B-spline functions, which are piecewise polynomial functions
with compact support. This kernel is particularly useful for problems requiring smooth
interpolation and approximation.

This kernel is particularly useful when you want your AI model to produce smooth outputs
and avoid abrupt changes in predictions, similar to how a skilled artist might draw a smooth
curve rather than connecting dots with straight lines.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BSplineKernel(Int32,)` | Initializes a new instance of the B-Spline kernel with optional parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BSplineBasis(Int32,)` | Calculates the value of the B-spline basis function of a given degree. |
| `Calculate(Vector<>,Vector<>)` | Calculates the B-Spline kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_degree` | The degree of the B-spline function used in the kernel calculation. |
| `_knotSpacing` | The spacing between knots in the B-spline function. |
| `_numOps` | Operations for performing numeric calculations with type T. |

