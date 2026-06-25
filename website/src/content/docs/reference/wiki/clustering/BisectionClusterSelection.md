---
title: "BisectionClusterSelection"
description: "Methods for selecting which cluster to bisect next."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Options`

Methods for selecting which cluster to bisect next.

## Fields

| Field | Summary |
|:-----|:--------|
| `HighestInertia` | Bisect the cluster with the highest inertia (sum of squared distances). |
| `Largest` | Always bisect the cluster with the most points. |
| `LargestDiameter` | Bisect the cluster with the largest diameter (max pairwise distance). |

