---
title: "ClusteringScores"
description: "Contains the results of cluster evaluation metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Contains the results of cluster evaluation metrics.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjustedRandIndex` | Adjusted Rand Index (-1 to 1, higher is better). |
| `CalinskiHarabasz` | Calinski-Harabasz Index (higher is better). |
| `DaviesBouldin` | Davies-Bouldin Index (lower is better). |
| `HasExternalMetrics` | Whether external metrics (ARI, NMI) are available. |
| `NormalizedMutualInformation` | Normalized Mutual Information (0 to 1, higher is better). |
| `Silhouette` | Silhouette Score (-1 to 1, higher is better). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a summary string of all metrics. |

