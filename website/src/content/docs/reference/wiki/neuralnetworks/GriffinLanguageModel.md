---
title: "GriffinLanguageModel<T>"
description: "Implements a Griffin language model: embedding + N RGLR blocks with local attention + layer norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Griffin language model: embedding + N RGLR blocks with local attention + layer norm + LM head.

## For Beginners

Griffin combines fast linear recurrence blocks with local sliding window
attention for a hybrid model that processes sequences efficiently while maintaining quality.

## How It Works

Griffin from Google DeepMind combines Real-Gated Linear Recurrence (RGLR) blocks with local
sliding window attention for a hybrid architecture achieving near-Transformer quality with
sub-quadratic complexity. Every Mth block uses local attention instead of RGLR.

**Reference:** De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Griffin blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

