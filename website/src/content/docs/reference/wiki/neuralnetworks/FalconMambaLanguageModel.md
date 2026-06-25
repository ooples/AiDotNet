---
title: "FalconMambaLanguageModel<T>"
description: "Implements a Falcon Mamba language model: embedding + N MambaBlock blocks + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Falcon Mamba language model: embedding + N MambaBlock blocks + RMS norm + LM head.

## For Beginners

Falcon Mamba is a 7-billion parameter language model that works
without attention mechanisms (unlike GPT or LLaMA). Instead, it uses the Mamba architecture
which processes sequences with constant memory, meaning it can handle infinitely long
conversations without slowing down. This makes it uniquely suited for applications requiring
very long context windows.

## How It Works

Falcon Mamba from TII (Technology Innovation Institute) is a pure Mamba-based language model
trained at 7B scale, achieving competitive results with Transformer-based models while
maintaining constant memory during generation regardless of sequence length.

**Reference:** Zuo et al., "Falcon Mamba: The First Competitive Attention-free 7B Language Model", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Falcon Mamba blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

