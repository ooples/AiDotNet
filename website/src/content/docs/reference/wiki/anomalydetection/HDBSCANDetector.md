---
title: "HDBSCANDetector<T>"
description: "Detects anomalies using HDBSCAN (Hierarchical DBSCAN)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.ClusterBased`

Detects anomalies using HDBSCAN (Hierarchical DBSCAN).

## For Beginners

HDBSCAN is an improved version of DBSCAN that automatically
finds clusters of varying densities without requiring an epsilon parameter.

## How It Works

The algorithm works by:

1. Compute core distances for each point
2. Build a mutual reachability graph
3. Construct a minimum spanning tree
4. Extract a cluster hierarchy
5. Points not in any cluster are anomalies

**When to use:**

- When clusters have varying densities
- When you don't want to tune epsilon
- Complex data structures with hierarchical cluster patterns

**Industry Standard Defaults:**

- Min cluster size: 5
- Min samples: 5
- Contamination: 0.1 (10%)

Reference: Campello, R., et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates." PAKDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HDBSCANDetector(Int32,Int32,Double,Int32)` | Creates a new HDBSCAN anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MinClusterSize` | Gets the minimum cluster size. |
| `MinSamples` | Gets the minimum number of samples for core points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

