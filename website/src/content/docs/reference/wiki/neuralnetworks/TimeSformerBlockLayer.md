---
title: "TimeSformerBlockLayer<T>"
description: "TimeSformer encoder block with divided space-time attention."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

TimeSformer encoder block with divided space-time attention.

## How It Works

Implements the Bertasius et al. TimeSformer block pattern: temporal attention is
applied across frames for each spatial patch, spatial attention is then applied
within each frame, and a position-wise FFN follows. Residual connections and
pre-layer-normalization match modern transformer training practice.

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>,Int32)` | Runs divided attention using the actual frame count from the video tokenizer. |

