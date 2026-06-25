---
title: "WCSS<T>"
description: "Within-Cluster Sum of Squares (WCSS) metric for evaluating cluster compactness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Within-Cluster Sum of Squares (WCSS) metric for evaluating cluster compactness.

## For Beginners

WCSS measures how "tight" your clusters are.

For each point:

- Find the center of its cluster
- Measure the distance squared
- Add up all these distances

Lower WCSS = Better clustering (tighter clusters)

Also called "inertia" in scikit-learn.

## How It Works

WCSS measures the total squared distance of each point to its cluster centroid.
Lower values indicate more compact clusters.

WCSS = sum over all clusters k of sum over all points i in k of ||x_i - c_k||^2

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WCSS` | Initializes a new WCSS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` |  |
| `ComputePerCluster(Matrix<>,Vector<>)` | Computes WCSS per cluster. |

