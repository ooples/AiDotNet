---
title: "BallTree<T>"
description: "Ball Tree for efficient nearest neighbor queries with arbitrary distance metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.SpatialIndex`

Ball Tree for efficient nearest neighbor queries with arbitrary distance metrics.

## For Beginners

A Ball Tree groups nearby points into "balls" (spheres).
Each ball contains a center point and a radius that encloses all points in that ball.

When searching for neighbors:

- If the query point is far from a ball's center (farther than the ball's radius

plus your search radius), you can skip the entire ball.

- This saves a lot of computation for large datasets.

Ball Trees work better than KD-Trees when:

- Your data has many dimensions (>20)
- You're using non-Euclidean distance metrics (like cosine distance)

## How It Works

A Ball Tree partitions space using hyperspheres (balls) rather than hyperplanes.
This makes it more effective than KD-Trees for high-dimensional data and
for non-Euclidean distance metrics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BallTree(IDistanceMetric<>,Int32)` | Initializes a new Ball Tree instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of points in the tree. |
| `Dimensions` | Gets the number of dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build(Matrix<>)` | Builds the Ball Tree from the given data. |
| `QueryKNearest(Vector<>,Int32)` | Finds the k nearest neighbors to the query point. |
| `QueryRadius(Vector<>,)` | Finds all points within the given radius of the query point. |

