---
title: "JambaLanguageModel<T>"
description: "Implements a Jamba language model: embedding + HybridBlockScheduler (Mamba + Attention) + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Jamba language model: embedding + HybridBlockScheduler (Mamba + Attention) + RMS norm + LM head.

## For Beginners

Jamba combines Mamba's efficient long-range processing with
Transformer attention's precise token interactions for the best of both worlds.

## How It Works

Jamba from AI21 Labs is a hybrid SSM-Attention model that interleaves Mamba blocks with
full attention blocks using the Jamba schedule pattern (every Nth block is attention).
This achieves strong quality by leveraging attention's exact retrieval with Mamba's efficient long-range processing.

**Reference:** Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Jamba blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

