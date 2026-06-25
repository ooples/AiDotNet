---
title: "IClusterMetric<T>"
description: "Interface for cluster evaluation metrics."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Clustering.Evaluation`

Interface for cluster evaluation metrics.

## For Beginners

Cluster metrics answer "How good is this clustering?"

Two main types:

1. Internal: Use data only (no "correct" answer needed)
- Silhouette Score: How similar are points to their own cluster?
- Davies-Bouldin: Are clusters compact and well-separated?
- Calinski-Harabasz: Ratio of between-cluster to within-cluster variance

2. External: Compare to known labels
- Adjusted Rand Index: Agreement with ground truth
- Normalized Mutual Information: Information shared with ground truth

## How It Works

Cluster metrics assess the quality of clustering results.
They can be internal (using only the data) or external
(comparing to ground truth labels).

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` | Computes the metric value. |

