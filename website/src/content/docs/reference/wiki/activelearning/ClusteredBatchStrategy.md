---
title: "ClusteredBatchStrategy<T, TInput, TOutput>"
description: "Clustering-based batch selection strategy using k-means clustering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Batch`

Clustering-based batch selection strategy using k-means clustering.

## For Beginners

This strategy first groups similar samples into clusters,
then selects the most informative sample from each cluster. This ensures the batch
covers different regions of the data space.

## How It Works

**How It Works:**

**Advantages:**

**Considerations:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteredBatchStrategy` | Initializes a new ClusteredBatchStrategy with default settings. |
| `ClusteredBatchStrategy(Int32,Int32,Double)` | Initializes a new ClusteredBatchStrategy with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiversityTradeoff` |  |
| `Name` |  |
| `NumClusters` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClusterSamples(IDataset<,,>)` |  |
| `ComputeDiversity(,)` |  |
| `SelectBatch(Int32[],Vector<>,IDataset<,,>,Int32)` |  |
| `SelectFromClusters(Int32[],Vector<>,Int32)` |  |

