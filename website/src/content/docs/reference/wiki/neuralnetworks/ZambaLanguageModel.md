---
title: "ZambaLanguageModel<T>"
description: "Implements a Zamba language model: embedding + HybridBlockScheduler (Mamba + shared attention) + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Zamba language model: embedding + HybridBlockScheduler (Mamba + shared attention) + RMS norm + LM head.

## For Beginners

Zamba uses mostly Mamba blocks with a single shared attention layer
reused at regular intervals, achieving strong quality with fewer parameters.

## How It Works

Zamba from Zyphra uses a hybrid architecture where Mamba blocks are the backbone and a single
shared attention layer is interleaved at regular intervals. The shared attention weights reduce
parameter count while retaining attention's retrieval capabilities.

**Reference:** Glorioso et al., "Zamba: A Compact 7B SSM Hybrid Model", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Zamba blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

