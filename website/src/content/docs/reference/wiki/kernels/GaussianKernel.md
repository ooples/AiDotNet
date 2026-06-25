---
title: "GaussianKernel<T>"
description: "Implements the Gaussian (Radial Basis Function) kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Gaussian (Radial Basis Function) kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Gaussian kernel is like a "similarity detector" that gives higher values when points are close
together and lower values when they're far apart.

## How It Works

The Gaussian kernel, also known as the Radial Basis Function (RBF) kernel, is one of the most widely used
kernel functions in machine learning. It measures similarity based on the Euclidean distance between points
and transforms this distance using an exponential function.

Think of the Gaussian kernel as a bell-shaped curve centered on each data point. When you compare two points,
the kernel value tells you how much their "bells" overlap. Points that are close together have a lot of
overlap (high similarity), while distant points have little overlap (low similarity).

This kernel is particularly popular because it works well for many different types of data and problems.
It's often a good first choice when you're not sure which kernel to use.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianKernel(Double)` | Initializes a new instance of the Gaussian kernel with an optional bandwidth parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Gaussian kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The bandwidth parameter that controls how quickly similarity decreases with distance. |

