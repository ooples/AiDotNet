---
title: "LaplacianKernel<T>"
description: "Implements the Laplacian kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Laplacian kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Laplacian kernel is like a "similarity detector" that gives higher values when points are close
together and lower values when they're far apart.

## How It Works

The Laplacian kernel is a radial basis function kernel that uses the Manhattan distance (L1 norm)
instead of the Euclidean distance (L2 norm) used by the Gaussian kernel. It's particularly useful
for data that has sparse features or when you want to be less sensitive to outliers.

Think of this kernel as a "distance translator" - it takes the distance between two points and
converts it into a similarity score between 0 and 1. Points that are identical get a score of 1,
while points that are very different get a score closer to 0.

What makes the Laplacian kernel special is how it measures distance - it uses what's called the
"Manhattan distance" or "city block distance." Imagine you're in a city with a grid of streets:
you can only travel along the streets (not diagonally through buildings). The Manhattan distance
is the total number of blocks you'd need to walk. This makes the Laplacian kernel less sensitive
to outliers compared to kernels that use the straight-line (Euclidean) distance.

The Laplacian kernel is often used in machine learning tasks like classification, regression,
and anomaly detection, especially when dealing with high-dimensional or sparse data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LaplacianKernel()` | Initializes a new instance of the Laplacian kernel with an optional bandwidth parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Laplacian kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The bandwidth parameter that controls how quickly similarity decreases with distance. |

