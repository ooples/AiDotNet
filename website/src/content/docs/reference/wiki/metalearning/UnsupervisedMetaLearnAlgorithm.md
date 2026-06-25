---
title: "UnsupervisedMetaLearnAlgorithm<T, TInput, TOutput>"
description: "Implementation of Unsupervised Meta-Learning (Hsu et al., 2019)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Unsupervised Meta-Learning (Hsu et al., 2019).

## How It Works

Unsupervised Meta-Learning constructs pseudo-tasks by clustering task gradient profiles
in a compressed space. Tasks assigned to the same cluster are treated as similar and
their gradients are reinforced, while cross-cluster gradients are dampened. This enables
meta-learning to discover task structure without explicit labels. A prediction consistency
regularization ensures that the adapted model maintains stable predictions.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_centroids` | Cluster centroids: NumClusters × ClusteringDim. |
| `_clusterCounts` | Per-cluster count for EMA weighting. |

