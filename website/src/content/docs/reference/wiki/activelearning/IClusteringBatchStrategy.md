---
title: "IClusteringBatchStrategy<T, TInput, TOutput>"
description: "Interface for clustering-based batch selection."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for clustering-based batch selection.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClusters` | Gets the number of clusters to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClusterSamples(IDataset<,,>)` | Clusters the samples in the unlabeled pool. |
| `SelectFromClusters(Int32[],Vector<>,Int32)` | Selects the most informative sample from each cluster. |

