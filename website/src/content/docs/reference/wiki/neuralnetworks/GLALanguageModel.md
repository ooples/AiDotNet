---
title: "GLALanguageModel<T>"
description: "Implements a full GLA (Gated Linear Attention) language model: embedding + N GLA blocks + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full GLA (Gated Linear Attention) language model: embedding + N GLA blocks + RMS norm + LM head.

## For Beginners

GLA is an efficient attention mechanism that uses gates to control
information flow, achieving Transformer-level quality with much lower computational cost.

## How It Works

GLA introduces hardware-efficient gated linear attention with data-dependent gating and
chunk-wise parallel computation. It matches Transformer quality with sub-quadratic complexity.

**Reference:** Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of GLA blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

