---
title: "GMeansOptions<T>"
description: "Configuration options for G-Means clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for G-Means clustering.

## For Beginners

G-Means splits clusters that aren't "bell-shaped".

The assumption: Good clusters should look like Gaussian (bell curve) distributions.

How it works:

1. Start with a few clusters (K-Means)
2. For each cluster, test if points form a bell curve
3. If not Gaussian, split the cluster into two
4. Repeat until all clusters pass the Gaussian test

The significance level controls sensitivity:

- Higher: More likely to split (more clusters)
- Lower: Less likely to split (fewer clusters)

## How It Works

G-Means extends K-Means by testing whether data in each cluster follows
a Gaussian distribution. If not, the cluster is split. It uses the
Anderson-Darling test for normality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GMeansOptions` | Initializes GMeansOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistanceMetric` | Gets or sets the distance metric. |
| `MaxClusters` | Gets or sets the maximum number of clusters. |
| `MinClusters` | Gets or sets the minimum number of clusters. |
| `SignificanceLevel` | Gets or sets the significance level for the Anderson-Darling test. |

