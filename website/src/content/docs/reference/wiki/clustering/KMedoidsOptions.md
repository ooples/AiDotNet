---
title: "KMedoidsOptions<T>"
description: "Configuration options for K-Medoids (PAM) clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for K-Medoids (PAM) clustering.

## For Beginners

K-Medoids uses real data points as cluster centers.

K-Means vs K-Medoids:

- K-Means: Centers are averages (may not be real points)
- K-Medoids: Centers are actual data points (medoids)

When to use K-Medoids:

- Data has outliers (medoids are more robust)
- Need interpretable cluster representatives
- Distance metric isn't Euclidean
- Categorical or mixed data

PAM (Partitioning Around Medoids) is the classic algorithm:

1. Initialize with random medoids
2. Assign points to nearest medoid
3. Try swapping medoids with non-medoids
4. Keep swaps that reduce total cost
5. Repeat until no improvement

## How It Works

K-Medoids is similar to K-Means but uses actual data points (medoids) as
cluster centers instead of means. This makes it more robust to outliers
and allows use of any distance metric.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KMedoidsOptions` | Initializes KMedoidsOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the algorithm variant. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `Init` | Gets or sets the initialization method. |
| `NumClusters` | Gets or sets the number of clusters. |

