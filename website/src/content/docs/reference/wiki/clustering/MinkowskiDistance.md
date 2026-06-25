---
title: "MinkowskiDistance<T>"
description: "Computes Minkowski (Lp) distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.DistanceMetrics`

Computes Minkowski (Lp) distance between vectors.

## For Beginners

Minkowski distance is a "tunable" distance metric.
By changing the parameter p, you get different behaviors:

- p = 1: Manhattan distance (city block)
- p = 2: Euclidean distance (straight line)
- p → ∞: Chebyshev distance (maximum difference)

Higher p values emphasize larger differences more. Lower p values treat
all differences more equally.

## How It Works

Minkowski distance is a generalization of both Euclidean (p=2) and Manhattan (p=1)
distances. Different values of p produce different distance behaviors.

Formula: d(a, b) = (sum(|a[i] - b[i]|^p))^(1/p)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MinkowskiDistance(Double)` | Initializes a new instance with the specified p value. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `P` | Gets the p parameter (order) of this Minkowski distance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |

