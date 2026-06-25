---
title: "ChebyshevDistance<T>"
description: "Computes Chebyshev (L∞) distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.DistanceMetrics`

Computes Chebyshev (L∞) distance between vectors.

## For Beginners

Chebyshev distance is the largest difference between
any pair of corresponding elements. It's like a "worst case" distance.

Example: The distance between points (0, 0) and (3, 4) is 4
(max(|3-0|, |4-0|) = max(3, 4) = 4)

Named after Pafnuty Chebyshev, it's useful when:

- You care about the maximum deviation in any single feature
- Movement is possible in all directions at once (like a king in chess)
- You need a robust metric that isn't sensitive to many small differences

## How It Works

Chebyshev distance, also called maximum metric or L-infinity distance, is the
maximum absolute difference along any dimension. It's the limit of Minkowski
distance as p approaches infinity.

Formula: d(a, b) = max(|a[i] - b[i]|)

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |

