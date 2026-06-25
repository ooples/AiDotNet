---
title: "Zamba2LanguageModel<T>"
description: "Implements a Zamba2 language model: embedding + HybridBlockScheduler (Mamba2 + shared attention with LoRA) + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Zamba2 language model: embedding + HybridBlockScheduler (Mamba2 + shared attention with LoRA) + RMS norm + LM head.

## For Beginners

Zamba2 upgrades Zamba with Mamba2 blocks and LoRA-adapted shared
attention layers for better efficiency and quality.

## How It Works

Zamba2 from Zyphra improves on Zamba by using Mamba2 blocks instead of Mamba1, adding multiple
shared attention layers with LoRA adapters for differentiation, and concatenating the original
shared attention output with the Mamba block output before each attention invocation.

**Reference:** Glorioso et al., "Zamba2-7B", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Zamba2 blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

