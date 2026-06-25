---
title: "HawkLanguageModel<T>"
description: "Implements a Hawk language model: embedding + N pure RGLR blocks + layer norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Hawk language model: embedding + N pure RGLR blocks + layer norm + LM head.

## For Beginners

Hawk is a pure-recurrent model that uses no attention at all,
giving strict O(n) complexity and O(1) memory per token during generation.

## How It Works

Hawk is the pure-recurrent variant from Google DeepMind (companion to Griffin), using only
Real-Gated Linear Recurrence blocks without any attention. This gives strict O(n) complexity
and O(1) memory per token during generation.

**Reference:** De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Hawk blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

