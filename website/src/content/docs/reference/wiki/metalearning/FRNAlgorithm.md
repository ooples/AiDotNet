---
title: "FRNAlgorithm<T, TInput, TOutput>"
description: "Implementation of FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021).

## For Beginners

FRN asks "which class can best explain this query?":

**The core idea:**
Instead of comparing features directly (distance), try to RECONSTRUCT the query's
features using each class's support features. The class that can best rebuild
the query's features is the most likely match.

**How reconstruction works:**
For class k with support features S_k:

1. Find optimal weights w* = argmin ||query - S_k @ w||^2 + lambda * ||w||^2
2. This is ridge regression: w* = (S_k^T @ S_k + lambda*I)^(-1) @ S_k^T @ query
3. Reconstruction: query_hat = S_k @ w*
4. Reconstruction error: ||query - query_hat||^2

**Why reconstruction is better than distance:**

- Distance: "How far is the query from the class center?"
- Reconstruction: "Can the class's patterns explain the query's patterns?"
- Reconstruction is more expressive because it uses MULTIPLE support examples

as building blocks, not just their mean.

**Example:**
Cat class has: tabby, persian, siamese support examples.
A new calico query is a MIX of tabby and siamese patterns.
Distance to mean might be far, but reconstruction from tabby + siamese is great.
FRN correctly classifies it as a cat.

## How It Works

FRN classifies queries by attempting to reconstruct each query's feature map
from the feature maps of each class's support examples. The class whose support
features best reconstruct the query is chosen as the predicted class.

**Algorithm - FRN:**

Reference: Wertheimer, D., Tang, L., & Hariharan, B. (2021).
Few-Shot Classification With Feature Map Reconstruction Networks. CVPR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FRNAlgorithm(FRNOptions<,,>)` | Initializes a new FRN meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeReconstructionError(Vector<>,List<Vector<>>)` | Computes the reconstruction error for a query given class support features. |
| `ComputeReconstructionWeights(Vector<>,Vector<>)` | Computes reconstruction-based classification weights for query features. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `SolveLinearSystemStatic(Double[0:,0:],Double[],Int32)` | Solves a linear system Ax = b using Gaussian elimination. |

