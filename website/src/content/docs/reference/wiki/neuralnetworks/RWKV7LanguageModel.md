---
title: "RWKV7LanguageModel<T>"
description: "Implements a full RWKV-7 \"Goose\" language model: token embedding + N RWKV7Blocks + RMS normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full RWKV-7 "Goose" language model: token embedding + N RWKV7Blocks + RMS normalization + LM head.

## For Beginners

RWKV-7 is the latest version of the RWKV architecture that
combines the best of RNNs and Transformers, achieving linear-time inference with
competitive quality to Transformer models.

## How It Works

RWKV-7 introduces dynamic state evolution with learnable transition matrices,
group normalization on WKV output, and SiLU channel mixing for improved training stability.

**Reference:** Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.

## Properties

| Property | Summary |
|:-----|:--------|
| `FFNMultiplier` | Gets the FFN dimension multiplier. |
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of RWKV-7 blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

