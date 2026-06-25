---
title: "IClustering<T>"
description: "Defines the common interface for all clustering algorithms in the AiDotNet library."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Clustering.Interfaces`

Defines the common interface for all clustering algorithms in the AiDotNet library.

## For Beginners

Clustering is about finding natural groups in data.

Unlike classification (where you have labeled examples to learn from), clustering
discovers patterns on its own. For example:

- Grouping customers by purchasing behavior
- Identifying topics in documents
- Segmenting images into regions
- Detecting anomalies (points that don't belong to any cluster)

The algorithm decides how many groups exist and which points belong together.

## How It Works

Clustering is an unsupervised machine learning technique that groups similar data points
together without prior knowledge of the categories. This interface extends IFullModel with
clustering-specific functionality.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusterCenters` | Gets the cluster centers (centroids) for centroid-based clustering algorithms. |
| `Inertia` | Gets the inertia (within-cluster sum of squares) for centroid-based algorithms. |
| `Labels` | Gets the cluster labels assigned to each training sample after fitting. |
| `NumClusters` | Gets the number of clusters found or specified. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitPredict(Matrix<>)` | Fits the clustering model and returns cluster labels in one operation. |
| `Transform(Matrix<>)` | Transforms data into cluster-distance space. |

