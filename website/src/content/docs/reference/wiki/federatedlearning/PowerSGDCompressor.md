---
title: "PowerSGDCompressor<T>"
description: "PowerSGD: low-rank gradient compression using randomized SVD approximation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

PowerSGD: low-rank gradient compression using randomized SVD approximation.

## For Beginners

Sending a full gradient vector (millions of values) is expensive.
PowerSGD compresses it by finding a low-rank approximation — like summarizing a book with
just the key plot points. A rank-4 approximation of a 1M-parameter gradient only needs to
send ~8K values (500x compression!).

## How It Works

**How it works:**

**Warm-start:** Reusing P/Q from previous rounds accelerates convergence because
the gradient subspace changes slowly between rounds.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PowerSGDCompressor(AdvancedCompressionOptions)` | Initializes a new instance of `PowerSGDCompressor`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(Tensor<>,Int32)` | Compresses a gradient tensor using low-rank PowerSGD approximation. |
| `Decompress(Double[0:,0:],Double[0:,0:],Int32,Int32,Int32)` | Decompresses a PowerSGD representation back to a gradient tensor. |
| `GetCompressionRatio(Int32)` | Gets the compression ratio achieved (compressed size / original size). |

