---
title: "PowerKernel<T>"
description: "Implements the Power kernel for measuring dissimilarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Power kernel for measuring dissimilarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures relationships between data points.
While most kernels measure similarity (higher values mean more similar points), the Power kernel is special
because it measures dissimilarity - higher values (less negative) mean the points are more different from each other.

## How It Works

The Power kernel is a negative distance-based kernel that measures how different two data points are,
rather than how similar they are. It's sometimes called the "negative distance kernel."

Think of the Power kernel as a "difference detector" that emphasizes how far apart points are.
This can be useful when you want your algorithm to focus on the differences between data points
rather than their similarities.

The formula for the Power kernel is:
k(x, y) = -||x - y||^d
where:

- x and y are the two data points being compared
- ||x - y|| is the Euclidean distance between them
- d is the degree parameter
- The negative sign makes this a dissimilarity measure

Common uses include:

- Clustering algorithms where distance is more important than similarity
- Outlier detection
- Applications where the magnitude of difference matters more than similarity patterns

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PowerKernel()` | Initializes a new instance of the Power kernel with an optional degree parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Power kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_degree` | The degree parameter that controls how the distance affects the kernel value. |
| `_numOps` | Operations for performing numeric calculations with type T. |

