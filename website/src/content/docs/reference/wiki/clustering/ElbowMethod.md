---
title: "ElbowMethod<T>"
description: "Elbow Method for determining the optimal number of clusters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Elbow Method for determining the optimal number of clusters.

## For Beginners

The Elbow Method finds where adding clusters stops helping.

As you increase K:

- K=1: All data in one cluster (high WCSS)
- K=2: Two clusters (lower WCSS)
- K=3: Three clusters (even lower WCSS)
- Eventually, improvements become tiny

The "elbow" is where the curve bends:

- Before elbow: Big improvements per cluster
- After elbow: Tiny improvements per cluster

The elbow point is often the best K!

## How It Works

The Elbow Method plots the within-cluster sum of squares (WCSS) against
the number of clusters. The optimal K is at the "elbow" of the curve where
adding more clusters provides diminishing returns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElbowMethod(Nullable<Int32>)` | Initializes a new ElbowMethod instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Int32,Int32)` | Computes WCSS for a range of cluster counts. |
| `DetectElbow(Int32[],Double[])` | Detects the elbow point using the Kneedle algorithm. |

