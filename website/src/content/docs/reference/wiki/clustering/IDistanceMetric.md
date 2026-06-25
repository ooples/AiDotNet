---
title: "IDistanceMetric<T>"
description: "Defines an interface for computing distance or similarity between vectors."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Clustering.Interfaces`

Defines an interface for computing distance or similarity between vectors.

## For Beginners

A distance metric tells us how to measure
"how far apart" two data points are.

Common examples:

- Euclidean: Straight-line distance (what you'd measure with a ruler)
- Manhattan: City-block distance (sum of differences along each axis)
- Cosine: Angle between vectors (useful for text/documents)

The choice of distance metric can significantly affect clustering results.

## How It Works

Distance metrics are fundamental to many clustering algorithms as they define
what it means for two points to be "similar" or "close together". Different
metrics are suitable for different types of data and applications.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this distance metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` | Computes the distance between two vectors. |
| `ComputeInline([],[],Int32)` | Computes the distance between two raw arrays without allocating Vector objects. |
| `ComputePairwise(Matrix<>)` | Computes the full pairwise distance matrix between all rows. |
| `ComputePairwise(Matrix<>,Matrix<>)` | Computes pairwise distances between rows of two different matrices. |
| `ComputeToAll(Vector<>,Matrix<>)` | Computes distances from a single point to all rows in a matrix. |

