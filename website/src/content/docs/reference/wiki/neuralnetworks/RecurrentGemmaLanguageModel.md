---
title: "RecurrentGemmaLanguageModel<T>"
description: "Implements a RecurrentGemma language model: embedding + N RGLR blocks + layer norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a RecurrentGemma language model: embedding + N RGLR blocks + layer norm + LM head.

## For Beginners

RecurrentGemma is Google's production model that uses recurrence instead of
attention, giving O(n) complexity and constant memory per token during text generation.

## How It Works

RecurrentGemma is Google's production recurrent language model based on the Griffin architecture.
It uses Real-Gated Linear Recurrence (RGLR) blocks for O(n) complexity and O(1) per-token generation.

**Reference:** Botev et al., "RecurrentGemma: Moving Past Transformers for Efficient Open Language Models", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of RecurrentGemma blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

