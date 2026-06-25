---
title: "TanimotoKernel<T>"
description: "Implements the Tanimoto kernel (also known as the Jaccard kernel) for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Tanimoto kernel (also known as the Jaccard kernel) for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Tanimoto kernel is special because it focuses on the overlap between features relative to their
combined magnitude. It's like measuring how much two sets have in common compared to their total size.

## How It Works

The Tanimoto kernel is a similarity measure that is particularly useful for binary data
and chemical fingerprints, but can be applied to any vector data. It measures the ratio
of the intersection to the union of the features.

Think of it like this: Imagine you have two shopping lists. The Tanimoto kernel would measure
how many items appear on both lists (the overlap) divided by the total number of unique items
across both lists. A value of 1 means the lists are identical, while a value close to 0 means
they have very little in common.

The formula for the Tanimoto kernel is:
k(x, y) = (x · y) / (||x||² + ||y||² - x · y)
where:

- x · y is the dot product between vectors x and y
- ||x||² is the squared norm of vector x (dot product of x with itself)
- ||y||² is the squared norm of vector y (dot product of y with itself)

Common uses include:

- Comparing chemical compounds (molecular fingerprints)
- Document similarity in text analysis
- Binary feature vectors in machine learning
- Recommendation systems

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TanimotoKernel` | Initializes a new instance of the Tanimoto kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Tanimoto kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |

