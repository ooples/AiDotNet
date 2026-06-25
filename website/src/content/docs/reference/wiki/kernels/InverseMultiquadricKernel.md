---
title: "InverseMultiquadricKernel<T>"
description: "Implements the Inverse Multiquadric kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Inverse Multiquadric kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Inverse Multiquadric kernel is like a "similarity detector" that gives higher values when points
are close together and lower values when they're far apart.

## How It Works

The Inverse Multiquadric kernel is a radial basis function kernel that measures similarity
based on the distance between points. Unlike the Gaussian kernel, it decreases more slowly
as points get farther apart, making it useful for data with long-range dependencies.

Think of this kernel as a "distance translator" - it takes the distance between two points and
converts it into a similarity score. Points that are close together get a similarity score close to 1,
while points that are far apart get a score closer to 0, but the decrease is more gradual than with
some other kernels.

This kernel has a parameter 'c' that controls how quickly the similarity decreases with distance.
A larger value of 'c' makes the kernel more "tolerant" of distance, meaning points can be farther
apart and still be considered somewhat similar.

The Inverse Multiquadric kernel is often used in machine learning tasks like regression, classification,
and interpolation, especially when you want to capture both local and more distant relationships in your data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InverseMultiquadricKernel()` | Initializes a new instance of the Inverse Multiquadric kernel with an optional shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Inverse Multiquadric kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_c` | The shape parameter that controls how quickly similarity decreases with distance. |
| `_numOps` | Operations for performing numeric calculations with type T. |

