---
title: "OnlineKMeansOptions<T>"
description: "Configuration options for Online/Streaming K-Means."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Online/Streaming K-Means.

## For Beginners

Regular clustering loads ALL data at once.
That's a problem when:

- Data is too big to fit in memory
- Data arrives continuously (streaming)
- You need real-time updates

Online K-Means solves this by:

- Processing one point (or small batch) at a time
- Incrementally updating cluster centers
- Never needing to see all data at once

The learning rate controls how much new data affects existing clusters.

## How It Works

Online K-Means processes data points one at a time or in small batches,
making it suitable for streaming data or datasets too large to fit in memory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineKMeansOptions` | Initializes OnlineKMeansOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecayLearningRate` | Gets or sets whether to decay the learning rate over time. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `LearningRate` | Gets or sets the learning rate for center updates. |
| `MinLearningRate` | Gets or sets the minimum learning rate when decaying. |
| `NumClusters` | Gets or sets the number of clusters. |

