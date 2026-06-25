---
title: "AffinityType"
description: "Types of affinity/similarity computation."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Options`

Types of affinity/similarity computation.

## Fields

| Field | Summary |
|:-----|:--------|
| `NearestNeighbors` | Nearest neighbors: Points are similar if they're nearest neighbors. |
| `Polynomial` | Polynomial kernel. |
| `Precomputed` | Precomputed: Affinity matrix is provided directly. |
| `RBF` | RBF (Radial Basis Function) kernel: exp(-gamma * \|\|x-y\|\|²). |
| `Sigmoid` | Sigmoid kernel. |

