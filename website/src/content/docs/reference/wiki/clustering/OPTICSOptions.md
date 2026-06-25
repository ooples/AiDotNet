---
title: "OPTICSOptions<T>"
description: "Configuration options for OPTICS clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for OPTICS clustering.

## For Beginners

OPTICS is like DBSCAN but smarter about finding clusters.

The problem with DBSCAN:

- You must choose epsilon carefully
- One epsilon might not work for all clusters

OPTICS solves this by:

- Computing "reachability distances" for all points
- Creating an ordering that reveals cluster structure
- Allowing you to extract clusters at different density levels

Think of it like hiking through mountains:

- Valleys are dense clusters
- Peaks are boundaries between clusters
- You can see the structure at different altitudes

## How It Works

OPTICS (Ordering Points To Identify the Clustering Structure) is a density-based
algorithm similar to DBSCAN but doesn't require a specific epsilon value upfront.
Instead, it creates an ordering of points and their reachability distances.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the neighbor finding algorithm. |
| `ClusterEpsilon` | Gets or sets the epsilon for DBSCAN-style extraction. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `ExtractionMethod` | Gets or sets the cluster extraction method. |
| `LeafSize` | Gets or sets the leaf size for tree algorithms. |
| `MaxEpsilon` | Gets or sets the maximum epsilon distance. |
| `MinClusterSizeFraction` | Gets or sets the minimum cluster size as fraction of total points. |
| `MinSamples` | Gets or sets the minimum number of samples in a neighborhood. |
| `PredecessorCorrection` | Gets or sets the predecessor correction setting. |
| `Xi` | Gets or sets the xi parameter for Xi extraction method. |

