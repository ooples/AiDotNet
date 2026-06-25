---
title: "LogKernel<T>"
description: "Implements the Log kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Log kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Log kernel is special because it uses logarithms to determine similarity.

## How It Works

The Log kernel is a kernel function that uses the negative logarithm of the distance between points
to measure similarity. It can be useful for certain types of data where similarity decreases
logarithmically with distance.

Think of the Log kernel as a "similarity detector" that is very sensitive to small distances but less
sensitive to large distances. When two points are very close together, a small change in distance makes
a big difference in similarity. But when two points are already far apart, even a large additional
distance doesn't change the similarity much.

This is similar to how we perceive sound volume: the difference between a whisper and normal speech
seems much greater than the difference between a shout and a very loud shout, even though the actual
increase in sound energy might be the same.

The Log kernel can be useful for data where small differences are very important when points are similar,
but less important when points are already quite different.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogKernel()` | Initializes a new instance of the Log kernel with an optional degree parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Log kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_degree` | Controls the power to which the logarithm of the distance is raised. |
| `_numOps` | Operations for performing numeric calculations with type T. |

