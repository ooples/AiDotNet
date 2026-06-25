---
title: "DenclueOptions<T>"
description: "Configuration options for DENCLUE clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for DENCLUE clustering.

## For Beginners

DENCLUE sees data as a "landscape" where
dense areas form "mountains" (attractors).

Imagine pouring water on a terrain:

- Water flows downhill to valleys
- In DENCLUE, points "flow uphill" to peaks
- Points reaching the same peak form a cluster

Key parameters:

- Bandwidth (h): Controls how wide the "mountains" are
- Small h: Many narrow peaks (more clusters)
- Large h: Few wide peaks (fewer clusters)
- MinDensity: Minimum density for a peak to be a cluster center

## How It Works

DENCLUE (DENsity-based CLUstEring) uses kernel density estimation
to find clusters. Points are attracted to density maxima using
gradient ascent, and clusters are formed by grouping points that
converge to the same attractor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenclueOptions` | Initializes DenclueOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttractorMergeThreshold` | Gets or sets the distance threshold for merging attractors. |
| `Bandwidth` | Gets or sets the bandwidth parameter for the Gaussian kernel. |
| `ConvergenceThreshold` | Gets or sets the convergence threshold for hill climbing. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `MinDensity` | Gets or sets the minimum density threshold for cluster attractors. |

