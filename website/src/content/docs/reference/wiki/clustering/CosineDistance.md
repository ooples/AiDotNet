---
title: "CosineDistance<T>"
description: "Computes Cosine distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.DistanceMetrics`

Computes Cosine distance between vectors.

## For Beginners

Cosine distance measures how different the "direction"
of two vectors is, ignoring how long they are.

Example: Vectors [1, 0] and [0, 1] are perpendicular (90°), so their cosine
similarity is 0 and distance is 1. Vectors [1, 2] and [2, 4] point in the
same direction, so their distance is 0.

Best for:

- Text documents (TF-IDF vectors)
- Any data where magnitude doesn't matter, only direction
- Sparse high-dimensional data

## How It Works

Cosine distance measures the angle between two vectors, ignoring their magnitudes.
It's derived from cosine similarity: distance = 1 - similarity.

Formula: d(a, b) = 1 - (a · b) / (||a|| × ||b||)
where a · b is the dot product and ||x|| is the L2 norm.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeSimilarity(Vector<>,Vector<>)` | Computes cosine similarity (1 - distance). |

