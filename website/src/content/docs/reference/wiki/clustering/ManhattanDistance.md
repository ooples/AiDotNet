---
title: "ManhattanDistance<T>"
description: "Computes Manhattan (L1) distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.DistanceMetrics`

Computes Manhattan (L1) distance between vectors.

## For Beginners

Manhattan distance is like walking along city blocks.
You can only move horizontally or vertically, not diagonally.

Example: The distance between points (0, 0) and (3, 4) is 7
(|3-0| + |4-0| = 3 + 4 = 7)

Good for:

- High-dimensional data where Euclidean distance loses meaning
- When features have different scales
- Sparse data

## How It Works

Manhattan distance, also called taxicab or city-block distance, is the sum of
absolute differences along each dimension. It's named after the grid-like
street layout of Manhattan.

Formula: d(a, b) = sum(|a[i] - b[i]|)

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeInline([],[],Int32)` |  |

