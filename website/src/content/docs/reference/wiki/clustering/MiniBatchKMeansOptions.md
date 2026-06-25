---
title: "MiniBatchKMeansOptions<T>"
description: "Configuration options specific to MiniBatch K-Means clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options specific to MiniBatch K-Means clustering.

## For Beginners

MiniBatch K-Means is like regular K-Means but faster.

Instead of using all data points in every step, it uses a random sample (mini-batch).
This makes it:

- Much faster for large datasets (millions of points)
- Uses less memory
- Produces slightly less optimal clusters (but usually very close)

Use MiniBatch K-Means when:

- Your dataset has more than ~10,000 points
- Speed is more important than perfect clustering
- You're doing online/streaming clustering

## How It Works

MiniBatch K-Means is a variant of KMeans that uses mini-batches (random samples)
to reduce computation time for large datasets while producing similar results.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the size of the mini-batches. |
| `InitMethod` | Gets or sets the initialization method for cluster centers. |
| `InitSize` | Gets or sets the number of random samples used to initialize centers. |
| `InitialCenters` | Gets or sets custom initial cluster centers. |
| `MaxNoImprovement` | Gets or sets the number of iterations with no improvement to wait before stopping. |
| `NumClusters` | Gets or sets the number of clusters to find. |
| `ReassignEmptyClusters` | Gets or sets whether to reassign empty clusters. |
| `ReassignmentRatio` | Gets or sets the fraction of centers that must be reassigned in each iteration. |

