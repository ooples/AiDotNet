---
title: "WaveKernel<T>"
description: "Implements the Wave kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Wave kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Wave kernel is special because it creates a pattern of similarity that rises and falls like a wave
as points get farther apart. This is different from most kernels where similarity only decreases with distance.

## How It Works

The Wave kernel is a specialized kernel function that produces wave-like patterns in the similarity
space. It is based on the sinc function (sin(x)/x) and creates oscillating similarity values as the
distance between points increases.

Think of it like this: If you drop two stones in a pond, the Wave kernel is like measuring how the
ripples from each stone interact with each other. Sometimes the ripples add up (high similarity) and
sometimes they cancel out (low similarity), creating a wave-like pattern.

The formula for the Wave kernel is:
k(x, y) = sin(||x-y||/s) / (||x-y||/s)
where:

- ||x-y|| is the Euclidean distance between vectors x and y
- s (sigma) is a parameter that controls the width of the waves

Common uses include:

- Signal processing applications
- Time series analysis
- Problems where periodic patterns are important
- Specialized machine learning tasks where oscillating similarity is beneficial

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveKernel()` | Initializes a new instance of the Wave kernel with the specified sigma parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Wave kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The sigma parameter that controls the width of the waves in the kernel. |

