---
title: "FairCutForest<T>"
description: "Implements Fair-Cut Forest (FCF) for anomaly detection with balanced tree construction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TreeBased`

Implements Fair-Cut Forest (FCF) for anomaly detection with balanced tree construction.

## For Beginners

Fair-Cut Forest improves on Isolation Forest by using balanced
tree construction. Instead of random splits, it selects splits that more evenly
divide the data, leading to more consistent isolation of anomalies.

## How It Works

The algorithm works by:

1. Building trees with "fair" splits that balance the data more evenly
2. Using importance-weighted feature selection based on data spread
3. Computing anomaly scores based on average path length to isolation

**When to use:**

- High-dimensional data where standard Isolation Forest struggles
- When you need more consistent anomaly rankings
- Datasets with features of varying importance

**Industry Standard Defaults:**

- Number of trees: 100
- Max samples: 256
- Contamination: 0.1 (10%)

Reference: Inspired by improvements to Isolation Forest for fair/balanced splits.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FairCutForest(Int32,Int32,Double,Int32)` | Creates a new Fair-Cut Forest anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSamples` | Gets the maximum samples used per tree. |
| `NumTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

