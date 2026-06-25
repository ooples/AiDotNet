---
title: "ClusteringMetrics<T>"
description: "Represents clustering quality metrics for evaluating the performance of clustering algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents clustering quality metrics for evaluating the performance of clustering algorithms.

## For Beginners

This class stores measurements that tell you how good your clustering is.

When you group data into clusters (like organizing customers into segments or grouping similar documents),
you need to know if the grouping makes sense. This class provides several scores that help answer questions like:

- Are items in the same cluster similar to each other?
- Are different clusters well-separated from each other?
- How does your clustering compare to known "ground truth" groupings?

The metrics included are:

- **Silhouette Score**: Measures how well each item fits in its cluster (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Measures cluster separation (higher is better)
- **Davies-Bouldin Index**: Measures cluster compactness and separation (lower is better)
- **Adjusted Rand Index**: Compares clustering to ground truth labels (-1 to 1, higher is better)

These metrics are automatically calculated during cross-validation when your model produces cluster labels.

## How It Works

This class encapsulates various metrics used to assess the quality of clustering results.
These metrics help determine how well data points are grouped into clusters and whether
the clustering algorithm has produced meaningful, well-separated groups.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteringMetrics` | Initializes a new instance of the ClusteringMetrics class with default values (all metrics null). |
| `ClusteringMetrics(,,,)` | Initializes a new instance of the ClusteringMetrics class with specified metric values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjustedRandIndex` | Gets or sets the Adjusted Rand Index comparing clustering to ground truth labels. |
| `CalinskiHarabaszIndex` | Gets or sets the Calinski-Harabasz Index for the clustering. |
| `DaviesBouldinIndex` | Gets or sets the Davies-Bouldin Index for the clustering. |
| `SilhouetteScore` | Gets or sets the Silhouette Score for the clustering. |

