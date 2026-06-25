---
title: "LinearKernel<T>"
description: "Implements the Linear kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Linear kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Linear kernel is the most basic type of kernel - it simply measures similarity as the dot product
between two data points.

## How It Works

The Linear kernel is the simplest kernel function, which computes the dot product between two vectors.
It's equivalent to performing linear regression or linear classification in the original feature space.

Think of the dot product as a way to measure how much two vectors "agree" with each other:

- If the vectors point in similar directions with similar magnitudes, the dot product will be large and positive.
- If the vectors point in opposite directions, the dot product will be negative.
- If the vectors are perpendicular (at right angles), the dot product will be zero.

The Linear kernel is useful when you believe your data can be separated by a straight line (in 2D),
a plane (in 3D), or a hyperplane (in higher dimensions). It's often a good first choice when you're
not sure which kernel to use, as it's simple and computationally efficient.

Unlike more complex kernels, the Linear kernel doesn't transform your data into a higher-dimensional space.
This means it works well when your data already has enough features to be linearly separable, but it might
not perform as well when your data requires more complex decision boundaries.

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Linear kernel value between two vectors. |

