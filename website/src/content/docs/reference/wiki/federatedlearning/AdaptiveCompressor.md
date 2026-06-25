---
title: "AdaptiveCompressor<T>"
description: "Adaptive compressor: dynamically adjusts compression ratio per client based on bandwidth, gradient importance, and staleness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Adaptive compressor: dynamically adjusts compression ratio per client based on bandwidth,
gradient importance, and staleness.

## For Beginners

In a real federation, clients have different network speeds.
A phone on 5G can send more data than one on 3G. Rather than using the same compression
for everyone, the adaptive compressor gives faster clients less compression (better accuracy)
and slower clients more compression (saves bandwidth). It also prioritizes clients whose
gradients carry more information.

## How It Works

**How it works:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveCompressor(AdvancedCompressionOptions)` | Initializes a new instance of `AdaptiveCompressor`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(Tensor<>,Int32,Int32,Double)` | Compresses a gradient using an adaptively determined top-k ratio. |
| `ComputeAdaptiveRatio(Int32,Tensor<>,Int32)` | Gets the current adaptive compression ratio for a client. |
| `RecordBandwidth(Int32,Double)` | Records a bandwidth measurement for a client. |

