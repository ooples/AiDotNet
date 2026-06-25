---
title: "PruningMask<T>"
description: "Binary mask for pruning neural network weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Pruning`

Binary mask for pruning neural network weights.

## For Beginners

A pruning mask is like a stencil for your neural network weights.

Imagine you have a grid of numbers (your neural network weights):

- Some numbers are important and should stay
- Some numbers can be removed to make the model smaller

The pruning mask marks which ones to keep (1) and which to remove (0).
When you apply the mask, all the marked weights become zero, effectively removing
those connections from your neural network.

This helps create smaller, faster models that still work well!

## How It Works

PruningMask represents a binary matrix where 1 indicates weights to keep and 0 indicates weights
to prune (set to zero). It provides methods to apply the mask to weight matrices and tensors,
compute sparsity levels, and combine multiple masks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PruningMask(Boolean[0:,0:])` | Initializes a pruning mask from a 2D boolean array (for matrices). |
| `PruningMask(Boolean[])` | Initializes a pruning mask from a 1D boolean array (for vectors). |
| `PruningMask(Int32,Int32)` | Initializes a new pruning mask with all ones (no pruning). |
| `PruningMask(Matrix<>)` | Initializes a pruning mask from an existing matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Pattern` | Gets the sparsity pattern type (unstructured for this implementation). |
| `Shape` | Gets the shape of the mask as [rows, columns]. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Matrix<>)` | Applies the mask to a weight matrix by element-wise multiplication. |
| `Apply(Tensor<>)` | Applies the mask to a weight tensor. |
| `Apply(Vector<>)` | Applies the mask to a weight vector. |
| `CombineWith(IPruningMask<>)` | Combines two masks using logical AND operation. |
| `GetKeptIndices` | Gets indices of non-zero (kept) elements. |
| `GetMaskData` | Gets the raw mask data as a flat array. |
| `GetPrunedIndices` | Gets indices of zero (pruned) elements. |
| `GetSparsity` | Calculates the sparsity level of the mask. |
| `MatrixToTensor(Matrix<>)` | Converts a matrix to a 2D tensor. |
| `TensorToMatrix(Tensor<>)` | Converts a 2D tensor to a matrix. |
| `UpdateMask(Array)` | Updates the mask from an Array (supports both 1D and 2D arrays). |
| `UpdateMask(Boolean[0:,0:])` | Updates the mask with new keep/prune decisions for 2D masks. |
| `UpdateMask(Boolean[])` | Updates the mask with new keep/prune decisions for 1D masks. |

