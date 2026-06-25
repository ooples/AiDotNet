---
title: "NeighborAlgorithm"
description: "Algorithms for computing nearest neighbors."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Options`

Algorithms for computing nearest neighbors.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically select based on data characteristics. |
| `BallTree` | Use Ball Tree (better for high dimensions or non-Euclidean metrics). |
| `BruteForce` | Brute force computation (guaranteed correct, O(n²) complexity). |
| `KDTree` | Use KD-Tree for efficient queries (best for low dimensions). |

