---
title: "LaplacianShotAlgorithm<T, TInput, TOutput>"
description: "Implementation of LaplacianShot (Laplacian Regularized Few-Shot Learning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of LaplacianShot (Laplacian Regularized Few-Shot Learning).

## For Beginners

LaplacianShot adds graph-based label propagation:

**Step 1: Initial classification (like SimpleShot)**
Classify each query by distance to support centroids.

**Step 2: Build similarity graph**
Connect each query to its k nearest neighbors (in feature space).

**Step 3: Label propagation**
Iteratively smooth predictions across the graph:

- If your neighbors are confidently "cat", you should also lean toward "cat"
- The Laplacian matrix encodes this smoothness constraint mathematically

**Why it helps:**
Query examples near the decision boundary get "pulled" toward the correct class
by their more confident neighbors, reducing boundary errors.

## How It Works

LaplacianShot augments nearest-centroid classification with Laplacian regularization
over a kNN graph of query features. This propagates labels from confident predictions
to uncertain ones based on feature similarity.

**Algorithm - LaplacianShot:**

Reference: Ziko, I., Dolz, J., Granger, E., & Ben Ayed, I. (2020).
Laplacian Regularized Few-Shot Learning. ICML 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LaplacianShotAlgorithm(LaplacianShotOptions<,,>)` | Initializes a new LaplacianShot meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using Laplacian-regularized nearest-centroid classification. |
| `LaplacianRefine(Vector<>,Vector<>)` | Applies Laplacian regularization to refine predictions using the query graph. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for LaplacianShot. |

