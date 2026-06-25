---
title: "IPruningMask<T>"
description: "Represents a binary mask for pruning weights in a neural network layer."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a binary mask for pruning weights in a neural network layer.

## For Beginners

Think of a pruning mask as a stencil or template.

Imagine you're painting a picture and want to cover certain areas:

- The mask has holes (1s) where paint should go through (weights to keep)
- The mask is solid (0s) where paint should be blocked (weights to prune/remove)

In neural networks:

- A pruning mask helps you selectively remove less important connections
- This makes your model smaller and faster without losing too much accuracy
- The mask can be applied to weight matrices to zero out pruned weights

## How It Works

A pruning mask is a binary matrix that determines which weights to keep (1) and which to remove (0)
during model compression. It enables selective removal of network parameters while maintaining the
ability to restore the network structure.

## Properties

| Property | Summary |
|:-----|:--------|
| `Pattern` | Gets the sparsity pattern type. |
| `Shape` | Gets the mask dimensions matching the weight matrix shape. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Matrix<>)` | Applies the mask to a weight matrix (element-wise multiplication). |
| `Apply(Tensor<>)` | Applies the mask to a weight tensor (for convolutional layers). |
| `Apply(Vector<>)` | Applies the mask to a vector. |
| `CombineWith(IPruningMask<>)` | Combines this mask with another mask (logical AND). |
| `GetKeptIndices` | Gets indices of non-zero (kept) elements. |
| `GetMaskData` | Gets the raw mask data as a flat array. |
| `GetPrunedIndices` | Gets indices of zero (pruned) elements. |
| `GetSparsity` | Gets the sparsity ratio (proportion of zeros). |
| `UpdateMask(Array)` | Updates the mask with new N-D keep/prune decisions. |
| `UpdateMask(Boolean[0:,0:])` | Updates the mask based on new pruning criteria. |
| `UpdateMask(Boolean[])` | Updates the mask with new keep/prune decisions. |

