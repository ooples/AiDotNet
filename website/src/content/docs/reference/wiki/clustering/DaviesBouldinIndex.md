---
title: "DaviesBouldinIndex<T>"
description: "Davies-Bouldin Index for evaluating cluster quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Davies-Bouldin Index for evaluating cluster quality.

## For Beginners

Davies-Bouldin measures cluster separation.

The idea:

- Good clusters are compact (small S)
- Good clusters are well-separated (large d)
- For each cluster, find the worst overlap with another
- Average these worst cases

Lower score = Better clustering!
(Unlike Silhouette where higher is better)

A score of 0 would be perfect separation.

## How It Works

The Davies-Bouldin Index measures the average similarity between
each cluster and its most similar cluster. Lower values indicate
better clustering (more compact and well-separated clusters).

For each cluster i:

- S(i) = average distance from points to cluster centroid
- d(i,j) = distance between cluster centroids
- R(i,j) = (S(i) + S(j)) / d(i,j)
- DB = (1/k) * sum(max_j(R(i,j)))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DaviesBouldinIndex(IDistanceMetric<>)` | Initializes a new DaviesBouldinIndex instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` |  |

