---
title: "CauchyKernel<T>"
description: "Implements the Cauchy kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Cauchy kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Cauchy kernel is a specialized similarity measure that works well when your data might contain
unusual or extreme values (outliers).

## How It Works

The Cauchy kernel is based on the Cauchy distribution from probability theory. It is a
long-tailed kernel that decreases more slowly than the Gaussian kernel, making it more
robust to outliers in the data.

Think of the Cauchy kernel as a "forgiving" similarity measure. When comparing two data points,
it doesn't penalize large differences as severely as other kernels might. This makes it useful
when you want your AI model to be less sensitive to occasional extreme values in your data,
similar to how a good teacher might not let one bad test score heavily impact a student's overall grade.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CauchyKernel()` | Initializes a new instance of the Cauchy kernel with an optional scaling parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Cauchy kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The scaling parameter that controls the width of the kernel. |

