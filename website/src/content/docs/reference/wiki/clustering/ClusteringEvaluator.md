---
title: "ClusteringEvaluator<T>"
description: "Comprehensive evaluator for clustering results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Comprehensive evaluator for clustering results.

## For Beginners

This is your "one-stop shop" for evaluating clusters.

Instead of calling each metric separately:

- Create a ClusteringEvaluator
- Call EvaluateAll() to get all metrics at once
- Compare different clusterings easily

The evaluator handles:

- Internal validity: How well-structured are the clusters?
- External validity: How well do clusters match known labels?
- Model selection: Which K is best? Which algorithm works best?

Use this when:

- Comparing different clustering algorithms
- Tuning parameters (like number of clusters)
- Validating clustering results

## How It Works

ClusteringEvaluator provides a unified interface to compute multiple
cluster validity indices and metrics. It supports both internal metrics
(using only the data) and external metrics (comparing to ground truth).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteringEvaluator` | Initializes a new ClusteringEvaluator with default metrics. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExternalMetric(IExternalClusterMetric<>)` | Adds a custom external metric to the evaluator. |
| `AddInternalMetric(IClusterMetric<>)` | Adds a custom internal metric to the evaluator. |
| `CompareClusterings(Matrix<>,List<Vector<>>,List<String>)` | Compares multiple clustering results and ranks them. |
| `EvaluateAll(Matrix<>,Vector<>,Vector<>)` | Evaluates all metrics for a clustering result. |
| `EvaluateExternal(Vector<>,Vector<>)` | Evaluates all external metrics for a clustering result. |
| `EvaluateInternal(Matrix<>,Vector<>)` | Evaluates all internal metrics for a clustering result. |
| `FindOptimalK(Matrix<>,Func<Int32,Vector<>>,ValueTuple<Int32,Int32>)` | Finds the optimal number of clusters using multiple criteria. |

