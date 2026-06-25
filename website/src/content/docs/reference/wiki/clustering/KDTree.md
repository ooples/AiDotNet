---
title: "KDTree<T>"
description: "K-dimensional tree for efficient nearest neighbor queries."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.SpatialIndex`

K-dimensional tree for efficient nearest neighbor queries.

## For Beginners

A KD-Tree is like a special way of organizing data points
so you can quickly find nearby points.

Imagine organizing a phone book not just by last name, but alternating between
first name and last name at each level. This lets you narrow down your search
very quickly.

KD-Trees are used in:

- DBSCAN clustering (to find points within epsilon radius)
- K-Nearest Neighbors (KNN) classification
- Range searches (find all points in a region)

## How It Works

KD-Tree is a space-partitioning data structure for organizing points in k-dimensional
space. It enables efficient nearest neighbor searches in O(log n) average case,
compared to O(n) for brute-force search.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KDTree(IDistanceMetric<>,Int32)` | Initializes a new KD-Tree instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of points in the tree. |
| `Dimensions` | Gets the number of dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build(Matrix<>)` | Builds the KD-Tree from the given data. |
| `QueryKNearest(Vector<>,Int32)` | Finds the k nearest neighbors to the query point. |
| `QueryRadius(Vector<>,)` | Finds all points within the given radius of the query point. |
| `QueryRadiusIndices(Vector<>,)` | Finds all points within the given squared radius of the query point. |

