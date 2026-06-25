---
title: "DeepEMDAlgorithm<T, TInput, TOutput>"
description: "Implementation of DeepEMD (Earth Mover's Distance for Few-Shot Learning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of DeepEMD (Earth Mover's Distance for Few-Shot Learning).

## For Beginners

DeepEMD measures similarity like a puzzle matching game:

**How it works:**

1. Break each example into local "parts" (features at different spatial positions)
2. For each query-support pair, find the optimal matching of parts
3. The Earth Mover's Distance = minimum total cost to match all parts
4. Classify by finding the support class with smallest EMD

**Analogy: Moving dirt piles**
Imagine two arrangements of dirt piles. EMD measures the minimum amount of
work (weight x distance) needed to reshape one arrangement into the other.
Small EMD = similar arrangements = similar examples.

**Why optimal transport?**

- Handles part-to-part correspondences (left wing of bird A matches right wing of bird B)
- Robust to spatial misalignment
- Captures structural similarity that global features miss

**The Sinkhorn algorithm:**
Computing exact EMD is expensive. Sinkhorn approximation adds entropy
regularization, making it fast and differentiable for gradient-based training.

## How It Works

DeepEMD computes optimal transport distances between sets of local features
from support and query examples. This captures fine-grained structural similarity
that simpler metrics (cosine, Euclidean) cannot represent.

**Algorithm - DeepEMD:**

Reference: Zhang, C., Cai, Y., Lin, G., & Shen, C. (2020).
DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance
and Structured Classifiers. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepEMDAlgorithm(DeepEMDOptions<,,>)` | Initializes a new DeepEMD meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using EMD-based nearest-class classification. |
| `ComputeEMDScore(Vector<>,Vector<>)` | Computes the approximate Earth Mover's Distance between two feature sets using the Sinkhorn algorithm. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for DeepEMD. |

