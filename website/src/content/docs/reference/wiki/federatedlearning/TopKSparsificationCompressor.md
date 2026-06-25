---
title: "TopKSparsificationCompressor<T>"
description: "Implements Top-k Sparsification — send only the k largest gradient elements."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Implements Top-k Sparsification — send only the k largest gradient elements.

## For Beginners

Most gradient values are small and contribute little to learning.
Top-k sparsification identifies the k largest values (by magnitude) and only sends those,
zeroing out the rest. With error feedback (accumulating the "residual" for next round),
this converges to the same solution as full gradient but with much less communication.

## How It Works

Algorithm:

Reference: Aji, A. & Heafield, K. (2017). "Sparse Communication for Distributed
Gradient Descent." EMNLP 2017.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TopKSparsificationCompressor(Double,Boolean)` | Creates a new Top-k sparsification compressor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressionRatio` | Gets the compression ratio (fraction of elements kept). |
| `UseErrorFeedback` | Gets whether error feedback is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateSparse(Dictionary<Int32,Dictionary<String,List<ValueTuple<Int32,>>>>,Dictionary<String,Int32>)` | Aggregates sparse gradients from multiple clients by summing overlapping entries. |
| `Compress(Dictionary<String,[]>,Int32)` | Compresses a gradient using top-k sparsification with optional error feedback. |
| `CompressSparse(Dictionary<String,[]>,Int32)` | Compresses a gradient to a sparse representation (index-value pairs), saving bandwidth. |
| `Decompress(Dictionary<String,List<ValueTuple<Int32,>>>,Dictionary<String,Int32>)` | Decompresses a sparse representation back to a dense gradient dictionary. |

