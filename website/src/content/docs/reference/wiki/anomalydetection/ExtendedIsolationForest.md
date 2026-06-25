---
title: "ExtendedIsolationForest<T>"
description: "Detects anomalies using Extended Isolation Forest (EIF)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TreeBased`

Detects anomalies using Extended Isolation Forest (EIF).

## For Beginners

Extended Isolation Forest improves on the original Isolation Forest by
using hyperplane cuts instead of axis-parallel cuts. This eliminates biases that occur when
anomalies align with the axes, making detection more robust.

## How It Works

The algorithm works by:

1. Build trees using random hyperplane cuts (instead of axis-parallel)
2. Each cut is defined by a random normal vector and intercept
3. Anomalies still have shorter path lengths on average

**When to use:**

- Same use cases as Isolation Forest
- When anomalies may align with coordinate axes
- When you need more robust splits

**Industry Standard Defaults:**

- Extension level: Full (use all dimensions)
- Number of trees: 100
- Max samples: 256
- Contamination: 0.1 (10%)

Reference: Hariri, S., Kind, M.C., Brunner, R.J. (2019). "Extended Isolation Forest." IEEE TKDE.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExtendedIsolationForest(Int32,Int32,Int32,Double,Int32)` | Creates a new Extended Isolation Forest anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExtensionLevel` | Gets the extension level (number of dimensions for hyperplane cuts). |
| `MaxSamples` | Gets the maximum samples per tree. |
| `NumTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

