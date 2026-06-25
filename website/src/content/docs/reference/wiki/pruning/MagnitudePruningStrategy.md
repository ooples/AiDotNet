---
title: "MagnitudePruningStrategy<T>"
description: "Prunes weights with smallest absolute values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Pruning`

Prunes weights with smallest absolute values.

## For Beginners

This strategy removes the weakest connections in your neural network.

Think of it like trimming weak branches from a tree:

- Thick, strong branches (large weight values) carry lots of nutrients and stay
- Thin, weak branches (small weight values) don't contribute much and get trimmed

In mathematical terms:

- Each weight gets an importance score equal to its absolute value |w|
- Weights with the smallest scores are pruned (set to zero)
- This is simple but surprisingly effective!

For example:

- A weight of 0.001 has low importance and might be pruned
- A weight of 0.9 has high importance and will likely be kept
- A weight of -0.8 has high importance too (|-0.8| = 0.8)

This technique can often remove 50-90% of weights with minimal accuracy loss!

## How It Works

Magnitude-based pruning is one of the simplest and most effective pruning strategies.
It removes weights with the smallest absolute values, based on the intuition that
weights with small magnitudes contribute less to the network's output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MagnitudePruningStrategy` | Initializes a new instance of MagnitudePruningStrategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsStructured` | Gets whether this is structured pruning (false for magnitude-based). |
| `Name` | Gets the name of this pruning strategy. |
| `RequiresGradients` | Gets whether this strategy requires gradients (false for magnitude-based). |
| `SupportedPatterns` | Gets supported sparsity patterns. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPruning(Matrix<>,IPruningMask<>)` | Applies the pruning mask to matrix weights in-place. |
| `ApplyPruning(Tensor<>,IPruningMask<>)` | Applies the pruning mask to tensor weights in-place. |
| `ApplyPruning(Vector<>,IPruningMask<>)` | Applies the pruning mask to vector weights in-place. |
| `ComputeImportanceScores(Matrix<>,Matrix<>)` | Computes importance scores as absolute values of weights for matrices. |
| `ComputeImportanceScores(Tensor<>,Tensor<>)` | Computes importance scores as absolute values of weights for tensors. |
| `ComputeImportanceScores(Vector<>,Vector<>)` | Computes importance scores as absolute values of weights for vectors. |
| `Create2to4Mask(Tensor<>)` | Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible). |
| `CreateMask(Matrix<>,Double)` | Creates a pruning mask for matrices by selecting the smallest magnitude weights to prune. |
| `CreateMask(Tensor<>,Double)` | Creates a pruning mask for tensors by selecting the smallest magnitude weights to prune. |
| `CreateMask(Vector<>,Double)` | Creates a pruning mask for vectors by selecting the smallest magnitude weights to prune. |
| `CreateNtoMMask(Tensor<>,Int32,Int32)` | Creates an N:M structured sparsity mask. |
| `ToSparseFormat(Tensor<>,SparseFormat)` | Converts pruned weights to sparse format for efficient storage. |

