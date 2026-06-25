---
title: "XMeansOptions<T>"
description: "Configuration options for X-Means clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for X-Means clustering.

## For Beginners

X-Means finds K automatically.

The problem with K-Means: You must choose K (number of clusters).
X-Means solves this by:

1. Start with a small K
2. Try splitting each cluster into two
3. Keep the split if it improves the model (using BIC)
4. Stop when no splits improve the model

BIC (Bayesian Information Criterion) balances:

- Model fit (how well clusters explain data)
- Model complexity (penalizes too many clusters)

## How It Works

X-Means extends K-Means by automatically determining the optimal number
of clusters using the Bayesian Information Criterion (BIC). It starts
with a minimum number of clusters and iteratively splits clusters until
BIC stops improving.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XMeansOptions` | Initializes XMeansOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Criterion` | Gets or sets the information criterion to use. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `MaxClusters` | Gets or sets the maximum number of clusters. |
| `MinClusters` | Gets or sets the minimum number of clusters. |

