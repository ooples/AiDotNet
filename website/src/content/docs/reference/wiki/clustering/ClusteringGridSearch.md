---
title: "ClusteringGridSearch<T>"
description: "Grid search for clustering hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.AutoML`

Grid search for clustering hyperparameter optimization.

## For Beginners

Grid Search tries every combination of parameters.

Example: For KMeans with K={2,3,4} and init={"random","kmeans++"}
Grid Search will try:

- K=2, init=random
- K=2, init=kmeans++
- K=3, init=random
- K=3, init=kmeans++
- K=4, init=random
- K=4, init=kmeans++

It evaluates each combination and returns the best one.

## How It Works

ClusteringGridSearch systematically searches through a specified
parameter grid to find the optimal hyperparameters for a given
clustering algorithm.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteringGridSearch(ClusteringMetricType)` | Initializes a new ClusteringGridSearch instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Search(Matrix<>,Func<Dictionary<String,Object>,IClustering<>>,Dictionary<String,Object[]>)` | Performs grid search over parameter combinations. |
| `SearchCV(Matrix<>,Func<Dictionary<String,Object>,IClustering<>>,Dictionary<String,Object[]>,Int32)` | Performs grid search with cross-validation for more robust results. |

