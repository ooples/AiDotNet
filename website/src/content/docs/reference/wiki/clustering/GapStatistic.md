---
title: "GapStatistic<T>"
description: "Gap Statistic for determining the optimal number of clusters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Gap Statistic for determining the optimal number of clusters.

## For Beginners

Gap Statistic finds the "right" number of clusters.

The idea:

1. Cluster the data with k=1, 2, 3, ... clusters
2. For each k, measure how "compact" the clusters are (WCSS)
3. Compare this to what you'd expect from random data
4. The best k is where real data is MUCH better than random

Think of it like:

- "At k=3, my clustering is 10x better than random"
- "At k=4, it's only 2x better"
- "So k=3 is probably right!"

## How It Works

The Gap Statistic compares the within-cluster dispersion of the data to that
expected under a null reference distribution (uniform random). The optimal K
is where the gap between observed and expected is largest.

Gap(k) = E*[log(W_k)] - log(W_k)
Where:

- W_k = within-cluster sum of squares for k clusters
- E*[log(W_k)] = expected value under null reference distribution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GapStatistic(Int32,Nullable<Int32>)` | Initializes a new GapStatistic instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Int32,Int32)` | Computes the Gap Statistic for a range of cluster counts. |

