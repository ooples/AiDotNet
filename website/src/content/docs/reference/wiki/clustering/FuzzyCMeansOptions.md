---
title: "FuzzyCMeansOptions<T>"
description: "Configuration options for Fuzzy C-Means clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Fuzzy C-Means clustering.

## For Beginners

Fuzzy C-Means allows points to be "partially" in clusters.

Regular K-Means: "This point belongs to Cluster A"
Fuzzy C-Means: "This point is 70% Cluster A, 25% Cluster B, 5% Cluster C"

This is useful when:

- Cluster boundaries are unclear
- Points naturally belong to multiple categories
- You want uncertainty information

The fuzziness parameter (m) controls how soft the clustering is:

- m close to 1: Hard clustering (like K-Means)
- m = 2: Standard fuzzy clustering (recommended)
- m > 2: Very soft, overlapping clusters

## How It Works

Fuzzy C-Means (FCM) is a soft clustering algorithm where each point can belong
to multiple clusters with varying degrees of membership. Unlike K-Means which
assigns each point to exactly one cluster, FCM assigns membership probabilities.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FuzzyCMeansOptions` | Initializes FuzzyCMeansOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistanceMetric` | Gets or sets the distance metric. |
| `NumClusters` | Gets or sets the number of clusters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fuzziness` | Gets or sets the fuzziness parameter (m). |

