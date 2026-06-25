---
title: "ConnectivityIndex<T>"
description: "Computes the Connectivity Index for cluster validity assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Computes the Connectivity Index for cluster validity assessment.

## For Beginners

Connectivity checks if neighbors are together.

For each point, look at its closest neighbors:

- 1st nearest neighbor: If in different cluster, add 1/1 = 1.0
- 2nd nearest neighbor: If in different cluster, add 1/2 = 0.5
- 3rd nearest neighbor: If in different cluster, add 1/3 = 0.33
- And so on...

The penalty is higher for closer neighbors being separated.

Interpretation:

- 0 = Perfect (all nearest neighbors in same cluster)
- Higher values = Worse (clusters split natural groupings)

Unlike other metrics:

- Lower is better (0 is ideal)
- No upper bound
- Intuitive: "Are nearby points kept together?"

## How It Works

The Connectivity Index measures the degree to which points are connected
to their neighbors within the same cluster. Lower values indicate better
clustering where nearby points are grouped together.

Formula: Connectivity = sum over all points i, over L nearest neighbors:
(1/j) if the j-th nearest neighbor is in a different cluster

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConnectivityIndex(Int32,IDistanceMetric<>)` | Initializes a new ConnectivityIndex instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HigherIsBetter` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` |  |

