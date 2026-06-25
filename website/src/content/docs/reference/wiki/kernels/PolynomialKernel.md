---
title: "PolynomialKernel<T>"
description: "Implements the Polynomial kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Polynomial kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Polynomial kernel is special because it can find complex patterns in your data by implicitly
mapping your data to a higher-dimensional space without actually performing the expensive calculations
that would normally be required.

## How It Works

The Polynomial kernel is a popular kernel function used in machine learning algorithms
like Support Vector Machines (SVMs) to find patterns in higher-dimensional spaces.

Think of the Polynomial kernel as a "pattern detector" that can identify more complex relationships
than simple linear patterns. For example, if you're trying to classify data that can't be separated
by a straight line, the Polynomial kernel can help find a curved boundary instead.

The formula for the Polynomial kernel is:
k(x, y) = (x·y + c)^d
where:

- x and y are the two data points being compared
- x·y is the dot product (a measure of how aligned the vectors are)
- c is a constant term (coef0) that influences the kernel's behavior
- d is the degree of the polynomial

Common uses include:

- Classification problems where data isn't linearly separable
- Natural language processing tasks
- Image recognition

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolynomialKernel(,)` | Initializes a new instance of the Polynomial kernel with optional degree and coefficient parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Polynomial kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coef0` | The constant coefficient added to the dot product before raising to the power of the degree. |
| `_degree` | The degree of the polynomial used in the kernel function. |
| `_numOps` | Operations for performing numeric calculations with type T. |

