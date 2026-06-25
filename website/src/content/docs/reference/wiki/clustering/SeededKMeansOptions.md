---
title: "SeededKMeansOptions<T>"
description: "Configuration options for Seeded K-Means."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Seeded K-Means.

## For Beginners

Instead of random initialization,
use data points you already know the labels for.

Example:

- You have 1000 customer profiles
- You've manually labeled 50 as "budget", "premium", or "enterprise"
- Seeded K-Means uses those 50 to initialize 3 clusters
- Then it clusters the remaining 950 automatically

This often produces better results than random initialization.

## How It Works

Seeded K-Means uses pre-labeled data points as initial cluster seeds.
The algorithm starts with these seeds and then proceeds like regular K-Means.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeededKMeansOptions` | Initializes SeededKMeansOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConstrainSeeds` | Gets or sets whether seeded points are constrained to their original cluster. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `NumClusters` | Gets or sets the number of clusters (inferred from seeds if not set). |
| `Seeds` | Gets or sets the labeled seed points. |

