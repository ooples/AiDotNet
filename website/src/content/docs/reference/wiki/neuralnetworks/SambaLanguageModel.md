---
title: "SambaLanguageModel<T>"
description: "Implements a Samba language model: embedding + HybridBlockScheduler (Mamba + sliding window attention) + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Samba language model: embedding + HybridBlockScheduler (Mamba + sliding window attention) + RMS norm + LM head.

## For Beginners

Samba alternates between Mamba blocks (for long-range context) and
sliding window attention (for precise local interactions) for efficient unlimited-context modeling.

## How It Works

Samba from Microsoft Research alternates Mamba blocks with sliding window attention blocks
in a regular pattern, combining Mamba's efficient long-range processing with local attention's
precise token interactions within a fixed window size.

**Reference:** Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Samba blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

