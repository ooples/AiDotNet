---
title: "FetchSGDCompressor<T>"
description: "Implements FetchSGD — Count-Sketch + Top-k hybrid compression for massive models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Implements FetchSGD — Count-Sketch + Top-k hybrid compression for massive models.

## For Beginners

FetchSGD combines two compression ideas: count-min sketches
(a probabilistic data structure) for aggregation and top-k for decompression. Each client
compresses their gradient into a small sketch (fixed-size regardless of model size). The
server merges sketches (just element-wise addition) and then recovers the top-k heavy hitters.
This is especially efficient for very large models (billions of parameters).

## How It Works

Algorithm:

Reference: Rothchild, D., et al. (2020). "FetchSGD: Communication-Efficient Federated
Learning with Sketching." ICML 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FetchSGDCompressor(Int32,Int32,Int32,Int32)` | Creates a new FetchSGD compressor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SketchCols` | Gets the sketch width. |
| `SketchRows` | Gets the sketch dimensions. |
| `TopK` | Gets the top-k value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `MergeSketches(IReadOnlyList<Double[0:,0:]>)` | Merges multiple sketches by element-wise addition. |
| `RecoverTopK(Double[0:,0:],Int32)` | Recovers the top-k heavy hitters from a merged sketch. |
| `Sketch([],Boolean)` | Compresses a flattened gradient into a count sketch with error feedback. |

