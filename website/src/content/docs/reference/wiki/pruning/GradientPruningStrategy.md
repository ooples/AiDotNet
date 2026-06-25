---
title: "GradientPruningStrategy<T>"
description: "Prunes weights based on gradient magnitude (sensitivity)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Pruning`

Prunes weights based on gradient magnitude (sensitivity).

## For Beginners

This strategy removes connections that don't learn much.

Think of it like identifying which team members contribute to a project:

- High gradient = This weight changes a lot during training, it's learning something important
- Low gradient = This weight barely changes, it's not contributing much to learning

The importance score is calculated as |weight × gradient|:

- If a weight is large BUT has tiny gradients, it might not be doing much
- If a weight is learning slowly (small gradient), removing it won't hurt performance

This is smarter than magnitude-based pruning because it considers learning dynamics,
not just weight size. However, it requires gradient information from training.

Example:

- Weight = 0.5, Gradient = 0.001 → Importance = |0.5 × 0.001| = 0.0005 (low, prune it)
- Weight = 0.3, Gradient = 0.9 → Importance = |0.3 × 0.9| = 0.27 (high, keep it)

## How It Works

Gradient-based pruning uses gradient information to determine weight importance.
Weights with small gradients have little impact on the loss function and can be safely removed.
This approach considers both the weight value and how much it affects learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientPruningStrategy` | Initializes a new instance of GradientPruningStrategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsStructured` | Gets whether this is structured pruning (false for gradient-based). |
| `Name` | Gets the name of this pruning strategy. |
| `RequiresGradients` | Gets whether this strategy requires gradients (true for gradient-based). |
| `SupportedPatterns` | Gets supported sparsity patterns. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPruning(Matrix<>,IPruningMask<>)` | Applies the pruning mask to weights in-place. |
| `ApplyPruning(Tensor<>,IPruningMask<>)` | Applies pruning mask to tensor weights in-place. |
| `ApplyPruning(Vector<>,IPruningMask<>)` | Applies pruning mask to vector weights in-place. |
| `ComputeImportanceScores(Matrix<>,Matrix<>)` | Computes importance scores as the product of weight magnitude and gradient magnitude. |
| `ComputeImportanceScores(Tensor<>,Tensor<>)` | Computes importance scores for tensor weights. |
| `ComputeImportanceScores(Vector<>,Vector<>)` | Computes importance scores for vector weights. |
| `Create2to4Mask(Tensor<>)` | Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible). |
| `CreateMask(Matrix<>,Double)` | Creates a pruning mask by selecting weights with lowest gradient-based importance. |
| `CreateMask(Tensor<>,Double)` | Creates a pruning mask for tensor weights based on target sparsity. |
| `CreateMask(Vector<>,Double)` | Creates a pruning mask for vector weights based on target sparsity. |
| `CreateNtoMMask(Tensor<>,Int32,Int32)` | Creates an N:M structured sparsity mask. |
| `ToSparseFormat(Tensor<>,SparseFormat)` | Converts pruned weights to sparse format for efficient storage. |
| `ToSparseFormat(Tensor<>,SparseFormat,Int32,Int32)` | Converts pruned weights to N:M structured sparse format for efficient storage. |

