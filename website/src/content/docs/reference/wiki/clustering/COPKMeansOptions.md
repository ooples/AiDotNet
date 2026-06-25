---
title: "COPKMeansOptions<T>"
description: "Configuration options for COP-KMeans (Constrained K-Means)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for COP-KMeans (Constrained K-Means).

## For Beginners

Sometimes you have partial knowledge about your data.

You might know:

- "These two customers bought the same product" (must-link)
- "These two users are different people" (cannot-link)

COP-KMeans uses this information to guide clustering:

- It won't separate must-linked pairs
- It won't group cannot-linked pairs

This is "semi-supervised" because you have some labels but not all.

## How It Works

COP-KMeans extends K-Means with pairwise constraints:

- Must-link: Two points must be in the same cluster
- Cannot-link: Two points must be in different clusters

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `COPKMeansOptions` | Initializes COPKMeansOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CannotLink` | Gets or sets the cannot-link constraints. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `MustLink` | Gets or sets the must-link constraints. |
| `NumClusters` | Gets or sets the number of clusters. |
| `UseTransitiveClosure` | Gets or sets whether to use transitive closure for must-link constraints. |

