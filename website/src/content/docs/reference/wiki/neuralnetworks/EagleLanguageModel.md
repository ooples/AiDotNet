---
title: "EagleLanguageModel<T>"
description: "Implements a full RWKV-5 \"Eagle\" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full RWKV-5 "Eagle" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head.

## For Beginners

Eagle (RWKV-5) is a language model that processes text like a
recurrent network (one token at a time) but can also be trained like a transformer (all tokens
in parallel). This gives it transformer-quality text generation with much lower memory usage
during inference, since it only needs to store a fixed-size state instead of the entire
conversation history.

## How It Works

This assembles the complete RWKV-5 "Eagle" architecture. Eagle introduces matrix-valued states
with multi-head attention, significantly improving upon RWKV-4's single-head scalar state design.

**Reference:** Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", 2024.
https://arxiv.org/abs/2404.05892

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Eagle blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

