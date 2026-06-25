---
title: "CircularKernel<T>"
description: "Implements the Circular kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Circular kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Circular kernel is special because it has a clear "cutoff point" - if two data points are too far
apart (farther than sigma), they're considered completely different (similarity = 0).

## How It Works

The Circular kernel is a compact (finite support) kernel based on the circular function from
statistics. It produces non-zero values only for points within a certain distance of each other,
defined by the sigma parameter.

Think of the Circular kernel like a neighborhood with a strict boundary. Points inside the neighborhood
have varying degrees of similarity based on how close they are to each other. But once you step outside
the neighborhood boundary (defined by sigma), there's no similarity at all.

This property makes the Circular kernel useful for problems where you want to focus only on local
patterns and completely ignore distant relationships.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CircularKernel()` | Initializes a new instance of the Circular kernel with an optional scaling parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Circular kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The scaling parameter that controls the radius of influence for the kernel. |

