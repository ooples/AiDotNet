---
title: "UnsupervisedMetaLearnOptions<T, TInput, TOutput>"
description: "Configuration options for Unsupervised Meta-Learning (Hsu et al., 2019)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Unsupervised Meta-Learning (Hsu et al., 2019).

## How It Works

Unsupervised meta-learning constructs pseudo-tasks by clustering gradients in a
low-dimensional space. Tasks within the same cluster share similar gradient structure
and are treated as the same "class" for self-supervised meta-training. Prediction
consistency regularization encourages stable cluster assignments.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusterUpdateRate` | EMA rate for updating cluster centroids. |
| `ClusteringDim` | Dimensionality of the compressed gradient space for clustering. |
| `ConsistencyWeight` | Weight on prediction consistency regularization between support and query adapted models. |
| `NumClusters` | Number of gradient clusters (pseudo-classes) for self-supervised task construction. |

