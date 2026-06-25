---
title: "FowlkesMallowsIndex<T>"
description: "Computes the Fowlkes-Mallows Index for cluster-label agreement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Computes the Fowlkes-Mallows Index for cluster-label agreement.

## For Beginners

The FM Index measures agreement between clusterings.

For every pair of points, ask two questions:

1. Are they in the same true class?
2. Are they in the same predicted cluster?

Possible outcomes:

- TP: Both same class AND same cluster (correct!)
- TN: Both different class AND different cluster (correct!)
- FP: Different class but same cluster (wrong!)
- FN: Same class but different cluster (wrong!)

FM Index = sqrt(Precision * Recall)

- Precision: Of pairs we grouped together, how many should be?
- Recall: Of pairs that should be together, how many did we group?

Range: 0 (no agreement) to 1 (perfect agreement)
Random clustering: FM ≈ sqrt(1/K) where K is number of clusters

## How It Works

The Fowlkes-Mallows Index (FMI) is the geometric mean of precision and recall
for pairs of points. It measures similarity between two clusterings and requires
ground truth labels.

Formula: FM = sqrt(TP / (TP + FP) * TP / (TP + FN))
where:

- TP = pairs correctly put in same cluster
- FP = pairs incorrectly put in same cluster
- FN = pairs incorrectly put in different clusters

## Properties

| Property | Summary |
|:-----|:--------|
| `HigherIsBetter` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Clustering#Evaluation#IExternalClusterMetric{T}#Compute(Vector<>,Vector<>)` |  |
| `Compute(Matrix<>,Vector<>)` |  |
| `ComputeWithTrueLabels(Matrix<>,Vector<>,Vector<>)` | Computes FM Index comparing predicted labels to true labels. |

