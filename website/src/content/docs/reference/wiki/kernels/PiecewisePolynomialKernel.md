---
title: "PiecewisePolynomialKernel<T>"
description: "Implements the Piecewise Polynomial kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Piecewise Polynomial kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Piecewise Polynomial kernel is special because it has a "cutoff distance" - if two points are
farther apart than this distance, the kernel says they have zero similarity.

## How It Works

The Piecewise Polynomial kernel is a compact kernel function that produces a similarity measure
that becomes exactly zero when points are far enough apart.

This property makes the Piecewise Polynomial kernel useful for:

- Speeding up calculations in large datasets (since many calculations become zero)
- Problems where you only want nearby points to influence each other
- Creating sparse matrices in kernel methods

The formula for this kernel is:
k(x, y) = (1 - ||x-y||/c)^(j+1) if ||x-y|| = c, and 0 otherwise
where:

- x and y are the two data points being compared
- ||x-y|| is the Euclidean distance between them
- c is the cutoff distance parameter
- j is the degree parameter

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PiecewisePolynomialKernel(Int32,)` | Initializes a new instance of the Piecewise Polynomial kernel with optional degree and cutoff parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Piecewise Polynomial kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_c` | The cutoff distance parameter that determines when the kernel value becomes zero. |
| `_degree` | The degree of the polynomial used in the kernel function. |
| `_numOps` | Operations for performing numeric calculations with type T. |

