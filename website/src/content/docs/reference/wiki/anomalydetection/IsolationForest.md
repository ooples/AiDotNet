---
title: "IsolationForest<T>"
description: "Implements the Isolation Forest algorithm for anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TreeBased`

Implements the Isolation Forest algorithm for anomaly detection.

## For Beginners

Isolation Forest is an efficient algorithm for detecting anomalies.
The key insight is that anomalies are "few and different" - they are easier to isolate
from the rest of the data.

## How It Works

The algorithm works by:

1. Building a "forest" of random isolation trees
2. Each tree randomly partitions the data by selecting random features and split points
3. Anomalies tend to be isolated in fewer steps (closer to the root of the tree)
4. Normal points require more splits to isolate (deeper in the tree)

**When to use:** Isolation Forest is particularly effective for:

- High-dimensional data
- Large datasets (it scales linearly)
- When you don't know the distribution of your data

**Industry Standard Defaults:**

- Number of trees: 100 (provides stable results)
- Max samples: 256 (balances speed and accuracy)
- Contamination: 0.1 (10%)

Reference: Liu, F. T., Ting, K. M., and Zhou, Z. H. (2008). "Isolation Forest."
In: ICDM '08. Eighth IEEE International Conference on Data Mining.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IsolationForest(Int32,Int32,Double,Int32)` | Creates a new Isolation Forest anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSamples` | Gets the number of samples used to train each tree. |
| `NumTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AveragePathLength(Int32)` | Calculates the average path length of unsuccessful search in a Binary Search Tree. |
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

