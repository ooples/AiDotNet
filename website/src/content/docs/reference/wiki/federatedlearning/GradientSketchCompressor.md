---
title: "GradientSketchCompressor<T>"
description: "Count Sketch-based gradient compression for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Count Sketch-based gradient compression for federated learning.

## For Beginners

A Count Sketch is a compact data structure that approximately stores
a large vector by hashing its elements into a smaller table. Think of it as a lossy compression
where you can recover the most important elements (top-k) but lose small values to hash collisions.

## How It Works

**How it works:**

**Compression ratio:** depth * width / gradient_size. With depth=5 and width=gradient_size/100,
this achieves ~20x compression.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientSketchCompressor(AdvancedCompressionOptions)` | Initializes a new instance of `GradientSketchCompressor`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(Tensor<>)` | Compresses a gradient tensor into a Count Sketch. |
| `Decompress(Double[0:,0:],Int32,Int32)` | Decompresses a Count Sketch back to a gradient tensor using median estimation. |
| `DecompressTopK(Double[0:,0:],Int32,Int32,Int32)` | Decompresses only the top-k largest elements from the sketch. |
| `GetCompressionRatio(Int32)` | Gets the compression ratio achieved (sketch size / original size). |

