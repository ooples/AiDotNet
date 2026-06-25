---
title: "StabilityValidation<T>"
description: "Stability-based validation for evaluating clustering quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Stability-based validation for evaluating clustering quality.

## For Beginners

Stability validation asks "Is this clustering reliable?"

If you cluster random samples of your data:

- High stability: Same clusters appear each time (good!)
- Low stability: Different clusters each time (concerning!)

A stable clustering is more trustworthy because it's not sensitive to
which exact points are included.

## How It Works

Stability validation measures how consistent clustering results are across
different random subsamples of the data. A good clustering should be stable

- similar subsamples should produce similar clusters.

Algorithm:

1. Generate multiple random subsamples of the data
2. Cluster each subsample
3. Measure agreement between clusterings on shared points
4. High stability indicates a robust clustering

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StabilityValidation(Int32,Double,Nullable<Int32>)` | Initializes a new StabilityValidation instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>,Int32)` | Evaluates clustering stability for a given number of clusters. |
| `EvaluateRange(Matrix<>,Int32,Int32)` | Evaluates stability across a range of cluster counts to find optimal K. |

