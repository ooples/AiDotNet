---
title: "IPruningStrategy<T>"
description: "Interface for pruning strategies that remove unimportant weights to create sparsity."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for pruning strategies that remove unimportant weights to create sparsity.

## For Beginners

Pruning removes unnecessary connections from neural networks.

Think of it like pruning a tree - you remove branches that don't contribute much:

- Magnitude pruning: Remove smallest weights
- Gradient pruning: Remove weights with smallest gradients (learning slowly)
- Structured pruning: Remove entire neurons/filters (cleaner architecture)
- Movement pruning: Remove weights that don't change during training
- Lottery ticket: Find sparse subnetworks that train well from scratch

Sparsity patterns:

- Unstructured: Random individual weights (flexible but needs sparse libraries)
- Structured: Entire rows/columns (actual speedup on any hardware)
- 2:4 Sparsity: 2 zeros per 4 elements (NVIDIA Ampere 2x speedup)
- N:M Sparsity: N zeros per M elements (customizable)

Pruning can remove 50-99% of weights with minimal accuracy loss!

## How It Works

Pruning strategies determine which weights to remove from a neural network to reduce size
and computational requirements. This interface supports all data types (Vector, Matrix, Tensor)
and multiple sparsity patterns including unstructured, structured, and hardware-optimized formats.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsStructured` | Gets whether this is structured pruning (removes entire rows/cols/filters). |
| `Name` | Gets the name of this pruning strategy. |
| `RequiresGradients` | Gets whether this strategy requires gradients. |
| `SupportedPatterns` | Gets supported sparsity patterns. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPruning(Matrix<>,IPruningMask<>)` | Applies pruning mask to matrix weights in-place. |
| `ApplyPruning(Tensor<>,IPruningMask<>)` | Applies pruning mask to tensor weights in-place. |
| `ApplyPruning(Vector<>,IPruningMask<>)` | Applies pruning mask to vector weights in-place. |
| `ComputeImportanceScores(Matrix<>,Matrix<>)` | Computes importance scores for matrix weights. |
| `ComputeImportanceScores(Tensor<>,Tensor<>)` | Computes importance scores for tensor weights. |
| `ComputeImportanceScores(Vector<>,Vector<>)` | Computes importance scores for vector weights. |
| `Create2to4Mask(Tensor<>)` | Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible). |
| `CreateMask(Matrix<>,Double)` | Creates a pruning mask for matrix weights based on target sparsity. |
| `CreateMask(Tensor<>,Double)` | Creates a pruning mask for tensor weights based on target sparsity. |
| `CreateMask(Vector<>,Double)` | Creates a pruning mask for vector weights based on target sparsity. |
| `CreateNtoMMask(Tensor<>,Int32,Int32)` | Creates an N:M structured sparsity mask. |
| `ToSparseFormat(Tensor<>,SparseFormat)` | Converts pruned weights to sparse format for efficient storage. |

