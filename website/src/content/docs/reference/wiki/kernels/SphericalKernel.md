---
title: "SphericalKernel<T>"
description: "Implements the Spherical kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Spherical kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Spherical kernel is special because it has a "limited range" - if two points are too far apart
(farther than the sigma parameter), the kernel says they have zero similarity. This is different from
many other kernels that might say points have a very small similarity even when they're very far apart.

## How It Works

The Spherical kernel is a type of compactly supported kernel, which means it becomes
exactly zero beyond a certain distance. This property makes it computationally efficient
for large datasets as it creates sparse matrices.

Think of it like this: The Spherical kernel creates a bubble of influence around each data point.
Points inside this bubble are considered similar (with similarity decreasing as distance increases),
while points outside the bubble are considered completely dissimilar.

The formula for the Spherical kernel is:
k(x, y) = 1.5 * (1 - ||x - y||/s) if ||x - y|| = s
k(x, y) = 0 if ||x - y|| > s
where:

- x and y are the two data points being compared
- ||x - y|| is the Euclidean distance between them
- s (sigma) is the radius parameter that determines the kernel's range

Common uses include:

- Spatial data analysis
- Geostatistics
- Large datasets where computational efficiency is important

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SphericalKernel()` | Initializes a new instance of the Spherical kernel with an optional radius parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Spherical kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The radius parameter that determines the range of influence for the kernel. |

