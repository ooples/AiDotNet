---
title: "CalinskiHarabaszIndex<T>"
description: "Calinski-Harabasz Index (Variance Ratio Criterion) for evaluating cluster quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Calinski-Harabasz Index (Variance Ratio Criterion) for evaluating cluster quality.

## For Beginners

Calinski-Harabasz measures variance ratio.

Good clustering has:

- High variance BETWEEN clusters (clusters are different)
- Low variance WITHIN clusters (clusters are tight)

The index is the ratio of these variances.
Higher score = Better clustering!

Think of it like:

- Numerator: How spread apart are the cluster centers?
- Denominator: How tight are points around their centers?
- We want big numerator, small denominator.

## How It Works

The Calinski-Harabasz Index is the ratio of between-cluster variance
to within-cluster variance. Higher values indicate better clustering.

CH = (BGS / (k-1)) / (WGS / (n-k))
Where:

- BGS = Between-Group Sum of Squares (cluster separation)
- WGS = Within-Group Sum of Squares (cluster compactness)
- k = number of clusters
- n = number of points

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CalinskiHarabaszIndex` | Initializes a new CalinskiHarabaszIndex instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` |  |

