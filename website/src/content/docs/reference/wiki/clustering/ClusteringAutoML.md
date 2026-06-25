---
title: "ClusteringAutoML<T>"
description: "Automatic machine learning for clustering - selects best algorithm and parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.AutoML`

Automatic machine learning for clustering - selects best algorithm and parameters.

## For Beginners

AutoML takes the guesswork out of clustering.

Instead of manually trying:

- Different algorithms (KMeans, DBSCAN, etc.)
- Different parameters (K values, epsilon, etc.)
- Evaluating results yourself

AutoML does it all automatically:

1. Tries multiple algorithms
2. Searches parameter spaces
3. Evaluates with multiple metrics
4. Returns the best solution

Just provide your data and AutoML finds the best clustering!

## How It Works

ClusteringAutoML provides automatic selection of clustering algorithms
and hyperparameters. It evaluates multiple algorithms with various
configurations and returns the best performing model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteringAutoML(ClusteringAutoMLOptions)` | Initializes a new ClusteringAutoML instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` | Runs automatic clustering algorithm and parameter selection. |

