---
title: "IMemoryBank<T>"
description: "Defines the contract for memory banks used in contrastive learning methods."
section: "API Reference"
---

`Interfaces` · `AiDotNet.SelfSupervisedLearning`

Defines the contract for memory banks used in contrastive learning methods.

## For Beginners

A memory bank is a queue that stores embeddings from previous
batches. This provides a large pool of negative samples without needing huge batch sizes.

## How It Works

**Why use a memory bank?**

**Used by:** MoCo, MoCo v2 (not MoCo v3, SimCLR, BYOL, or SimSiam)

**Example usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Capacity` | Gets the maximum capacity of the memory bank (queue size). |
| `CurrentSize` | Gets the current number of stored embeddings. |
| `EmbeddingDimension` | Gets the embedding dimension of stored vectors. |
| `IsFull` | Gets whether the memory bank is full (has reached capacity). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears all stored embeddings and resets the memory bank. |
| `Enqueue(Tensor<>)` | Adds new embeddings to the memory bank (FIFO queue). |
| `GetAll` | Gets all stored embeddings for use as negative samples. |
| `Sample(Int32)` | Gets a random subset of stored embeddings. |
| `UpdateWithMomentum(Int32[],Tensor<>,Double)` | Updates embeddings by averaging with new values (for soft updates). |

