---
title: "KairosMultiSizePatchLayer<T>"
description: "Kairos Mixture-of-Size patch embedder: emits N parallel patch-size paths through the SAME transformer backbone shape (numPatches varies per path but hiddenDim is fixed), then combines them via a learned router that weights each path per-inp…"
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Kairos Mixture-of-Size patch embedder: emits N parallel patch-size paths through the
SAME transformer backbone shape (numPatches varies per path but hiddenDim is fixed),
then combines them via a learned router that weights each path per-input. This is the
Mixture-of-Size analog of Mixture-of-Experts: the "experts" are alternative
tokenization granularities rather than alternative FFNs.

## How It Works

Forward:

This layer outputs a single summarized [B, hiddenDim] tensor per input, ready to feed
a downstream transformer stack that treats each input as a single token (since
multi-size pooling has already collapsed the sequence). For the full per-patch
sequence interpretation, the caller should use only one patch size.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KairosMultiSizePatchLayer(Int32,Int32,Int32[])` | Initializes a new `KairosMultiSizePatchLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Persists the constructor arguments so the deserializer can rebuild this layer at the same shape. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

