---
title: "AdditiveChiSquaredKernel<T>"
description: "Implements the Additive Chi-Squared kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Additive Chi-Squared kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a way to measure how similar two data points are to each other.
Think of it like a special ruler that can measure distance in complex data spaces. The Additive 
Chi-Squared kernel is particularly good at comparing data that represents counts or frequencies 
(like how many times words appear in documents, or how many pixels of each color appear in images).

## How It Works

The Additive Chi-Squared kernel is a variation of the Chi-Squared distance measure that is
particularly useful for histogram comparison in image recognition, document classification,
and other applications where data is represented as frequency distributions.

The kernel is defined as K(x,y) = -log(1 + S[(x_i - y_i)²/(x_i + y_i)]) for all dimensions i.

Unlike regular distance measures where smaller values mean "closer" (more similar), kernel functions
typically return larger values for more similar items. This kernel transforms the Chi-Squared distance
so that more similar items have higher values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdditiveChiSquaredKernel` | Initializes a new instance of the Additive Chi-Squared kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Additive Chi-Squared kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |

