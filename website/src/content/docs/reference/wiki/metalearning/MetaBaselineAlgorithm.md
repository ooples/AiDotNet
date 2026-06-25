---
title: "MetaBaselineAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-Baseline (simple pre-train then meta-train with cosine classifier)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-Baseline (simple pre-train then meta-train with cosine classifier).

## For Beginners

Meta-Baseline shows that the simplest approach can be the best:

1. Train normally on many classes to get good features
2. Switch to episodic training with cosine-distance centroids
3. At test time, classify by nearest centroid (cosine distance)

## How It Works

Meta-Baseline trains a feature extractor with standard classification, then fine-tunes
with episodic training using cosine similarity nearest-centroid classification.

Reference: Chen, Y., Liu, Z., Xu, H., Darrell, T., & Wang, X. (2021).
Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning. ICLR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaBaselineAlgorithm(MetaBaselineOptions<,,>)` | Initializes a new Meta-Baseline meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |
| `NormalizeVector(Vector<>)` | L2-normalizes a feature vector for cosine similarity computation. |

