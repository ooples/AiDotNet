---
title: "EntmaxAttention<T>"
description: "Entmax sparse attention function for NODE architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Entmax sparse attention function for NODE architecture.

## For Beginners

Entmax is like softmax but can completely ignore some inputs:

- Softmax: Every input gets some attention (even if tiny)
- Entmax: Unimportant inputs get exactly zero attention

This helps the model focus better and makes the attention patterns easier to interpret.

## How It Works

Entmax is a sparse alternative to softmax that can produce exact zeros in the output
distribution. This is useful for attention mechanisms where you want to focus on
only a few important elements.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntmaxAttention(Double)` | Initializes entmax attention. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Applies entmax to convert scores to a sparse probability distribution. |

