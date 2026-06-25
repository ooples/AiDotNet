---
title: "ProbabilisticKernel<T>"
description: "Implements the Probabilistic kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Probabilistic kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Probabilistic kernel is special because it considers both:

1. The angle between data points (like cosine similarity)
2. The difference in their magnitudes (like Gaussian kernel)

## How It Works

The Probabilistic kernel combines aspects of both the cosine similarity and the Gaussian kernel,
making it useful for capturing both directional and magnitude-based relationships between data points.

Think of the Probabilistic kernel as a "smart similarity detector" that can tell if two data points
are pointing in similar directions AND have similar sizes. This is particularly useful when both
the direction and magnitude of your data are important.

The formula for the Probabilistic kernel is:
k(x, y) = (x·y / v(||x||²·||y||²)) · exp(-||x||² - ||y||²)²/(2s²))
where:

- x and y are the two data points being compared
- x·y is the dot product (a measure of how aligned the vectors are)
- ||x|| and ||y|| are the magnitudes (lengths) of the vectors
- s (sigma) is a parameter that controls sensitivity to magnitude differences

Common uses include:

- Text classification where both word presence and frequency matter
- Image recognition where both pattern and intensity are important
- Any application where both direction and magnitude of data provide useful information

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProbabilisticKernel()` | Initializes a new instance of the Probabilistic kernel with an optional sigma parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Probabilistic kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The sigma parameter that controls sensitivity to differences in vector magnitudes. |

