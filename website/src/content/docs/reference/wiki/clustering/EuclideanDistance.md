---
title: "EuclideanDistance<T>"
description: "Computes Euclidean (L2) distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.DistanceMetrics`

Computes Euclidean (L2) distance between vectors.

## For Beginners

Euclidean distance is what you'd measure with a ruler
in a straight line between two points. It's the "as the crow flies" distance.

Example: The distance between points (0, 0) and (3, 4) is 5
(using the Pythagorean theorem: sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5)

## How It Works

Euclidean distance is the straight-line distance between two points in Euclidean space.
It is the most commonly used distance metric for clustering.

Formula: d(a, b) = sqrt(sum((a[i] - b[i])^2))

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeSquared(Vector<>,Vector<>)` | Computes the squared Euclidean distance (avoids square root for efficiency). |

