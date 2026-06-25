---
title: "ClusterBasedSplitter<T>"
description: "Cluster-based splitter that divides data by similarity clusters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized`

Cluster-based splitter that divides data by similarity clusters.

## For Beginners

This splitter groups similar samples together using clustering,
then assigns entire clusters to train or test. This ensures the test set contains
samples that are meaningfully different from training.

## How It Works

**When to Use:**

- When you want to test generalization to truly different data
- To avoid having very similar samples in both train and test
- For harder, more realistic model evaluation

**Note:** This splitter expects cluster assignments to be provided.
Run clustering (K-means, DBSCAN, etc.) beforehand and pass the cluster labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusterBasedSplitter(Int32[],Double,Boolean,Int32)` | Creates a cluster-based splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

