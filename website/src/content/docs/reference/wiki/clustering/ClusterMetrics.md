---
title: "ClusterMetrics<T>"
description: "Convenience class for computing multiple cluster evaluation metrics at once."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Convenience class for computing multiple cluster evaluation metrics at once.

## For Beginners

Use this class to evaluate your clustering results.

Example usage:

If you have ground truth labels:

## How It Works

This class provides a simple interface to compute multiple clustering
metrics with a single call. It returns a ClusteringScores object
containing all computed values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusterMetrics(IDistanceMetric<>)` | Initializes a new ClusterMetrics instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>,Vector<>)` | Evaluates clustering using internal metrics only. |
| `Evaluate(Matrix<>,Vector<>,Vector<>)` | Evaluates clustering using both internal and external metrics. |
| `EvaluateExternal(Vector<>,Vector<>)` | Computes only external metrics (when data is not available). |

