---
title: "MemoryBank<T>"
description: "FIFO memory queue for storing embeddings in contrastive learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

FIFO memory queue for storing embeddings in contrastive learning.

## For Beginners

A memory bank is a queue that stores embeddings from previous
batches. This provides a large pool of negative samples for contrastive learning without
requiring huge batch sizes.

## How It Works

**How it works:**

**Example usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryBank(Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the MemoryBank class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Capacity` |  |
| `CurrentSize` |  |
| `EmbeddingDimension` |  |
| `IsFull` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` |  |
| `Enqueue(Tensor<>)` |  |
| `GetAll` |  |
| `GetAt(Int32)` | Gets the embedding at a specific index. |
| `Sample(Int32)` |  |
| `SetAt(Int32,Tensor<>)` | Sets the embedding at a specific index. |
| `UpdateWithMomentum(Int32[],Tensor<>,Double)` |  |

