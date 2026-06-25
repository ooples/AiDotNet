---
title: "SCiForest<T>"
description: "Detects anomalies using SCiForest (Sparse Clustering-Integrated Isolation Forest)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TreeBased`

Detects anomalies using SCiForest (Sparse Clustering-Integrated Isolation Forest).

## For Beginners

SCiForest improves on Isolation Forest by using sparse splits that
combine multiple features. This makes it better at detecting anomalies in subspaces and
handles high-dimensional data more effectively.

## How It Works

The algorithm works by:

1. Build trees using sparse random projections for splits
2. Each split uses a weighted combination of features
3. Anomalies have shorter average path lengths

**When to use:**

- High-dimensional data
- When anomalies hide in subspaces
- When standard Isolation Forest doesn't perform well

**Industry Standard Defaults:**

- Number of trees: 100
- Max samples: 256
- Sparsity: 0.2 (20% of features per split)
- Contamination: 0.1 (10%)

Reference: Liu, F.T., Ting, K.M., Zhou, Z.H. (2012). "Isolation-Based Anomaly Detection."
TKDD 6(1), with sparse extensions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SCiForest(Int32,Int32,Double,Double,Int32)` | Creates a new SCiForest anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSamples` | Gets the maximum samples per tree. |
| `NumTrees` | Gets the number of trees in the forest. |
| `Sparsity` | Gets the sparsity level for projections. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

