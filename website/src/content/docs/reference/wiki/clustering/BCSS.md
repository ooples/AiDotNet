---
title: "BCSS<T>"
description: "Between-Cluster Sum of Squares (BCSS) metric for evaluating cluster separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Between-Cluster Sum of Squares (BCSS) metric for evaluating cluster separation.

## For Beginners

BCSS measures how "spread apart" your cluster centers are.

- Find the overall center of all data
- For each cluster, measure distance from cluster center to overall center
- Weight by cluster size

Higher BCSS = Better clustering (clusters are more separated)

Total variance = WCSS + BCSS
Good clustering has low WCSS (tight) and high BCSS (separated).

## How It Works

BCSS measures the weighted sum of squared distances between cluster centroids
and the overall data centroid. Higher values indicate better separated clusters.

BCSS = sum over all clusters k of n_k * ||c_k - c_global||^2

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BCSS` | Initializes a new BCSS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` |  |

