---
title: "MahalanobisDistance<T>"
description: "Computes Mahalanobis distance between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.DistanceMetrics`

Computes Mahalanobis distance between vectors.

## For Beginners

Mahalanobis distance is "smart" about how features
relate to each other.

Example: If height and weight are correlated (tall people tend to weigh more),
Euclidean distance treats them as independent, but Mahalanobis distance
accounts for this relationship.

It's like asking "how many standard deviations away is this point?"
taking into account that the data might be stretched or tilted.

Best for:

- Detecting outliers
- Gaussian Mixture Models
- When features have different scales or correlations

## How It Works

Mahalanobis distance accounts for correlations between variables and scales
by variance. It's the standard metric used in Gaussian Mixture Models (GMM).
When the covariance matrix is the identity, it reduces to Euclidean distance.

Formula: d(a, b) = sqrt((a - b)^T × Σ^(-1) × (a - b))
where Σ is the covariance matrix.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MahalanobisDistance` | Initializes a new instance without a covariance matrix. |
| `MahalanobisDistance(Matrix<>)` | Initializes a new instance with a precomputed inverse covariance matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InverseCovarianceMatrix` | Gets or sets the inverse covariance matrix used for distance computation. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `FitFromData(Matrix<>)` | Computes the Mahalanobis distance from data, estimating the covariance matrix. |

