---
title: "CosineKernel<T>"
description: "Cosine Similarity Kernel that measures angular distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Cosine Similarity Kernel that measures angular distance between vectors.

## For Beginners

The Cosine Kernel computes the cosine of the angle between two vectors.
It measures similarity based on direction rather than magnitude.

In mathematical terms:
k(x, x') = (x · x') / (||x|| × ||x'||)

Where:

- x · x' is the dot product
- ||x|| is the Euclidean norm (length) of x

Properties:

- Returns values in [-1, 1]
- k(x, x') = 1 when vectors point in the same direction
- k(x, x') = 0 when vectors are orthogonal
- k(x, x') = -1 when vectors point in opposite directions

This kernel is particularly useful for:

- Text classification (TF-IDF vectors)
- Document similarity
- Any application where magnitude doesn't matter, only direction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineKernel(Double)` | Initializes a new Cosine Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputScale` | Gets the output scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the cosine similarity between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_outputScale` | Optional output scale factor. |

