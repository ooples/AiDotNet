---
title: "MultiquadricKernel<T>"
description: "Implements the Multiquadric kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Multiquadric kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Multiquadric kernel is particularly useful for spatial data and regression problems.

## How It Works

The Multiquadric kernel is a popular kernel function used in machine learning and spatial analysis
to measure the similarity between data points.

Think of the Multiquadric kernel as a "similarity detector" that works well for many types of data.
Unlike some other kernels, the Multiquadric kernel increases with distance, which gives it some
unique properties that can be useful for certain types of problems.

The formula for the Multiquadric kernel is:
k(x, y) = v(||x - y||² + c²)
where:

- x and y are the two data points being compared
- ||x - y|| is the Euclidean distance between them
- c is a parameter that controls the kernel's behavior

This kernel is often used in radial basis function networks, spatial interpolation,
and various machine learning algorithms where you need to measure similarity between points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiquadricKernel()` | Initializes a new instance of the Multiquadric kernel with an optional shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Multiquadric kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_c` | The shape parameter that controls the behavior of the kernel function. |
| `_numOps` | Operations for performing numeric calculations with type T. |

