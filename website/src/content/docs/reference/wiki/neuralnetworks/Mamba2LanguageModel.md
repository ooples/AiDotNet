---
title: "Mamba2LanguageModel<T>"
description: "Implements a Mamba-2 language model: token embedding + N Mamba2Blocks + layer normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Mamba-2 language model: token embedding + N Mamba2Blocks + layer normalization + LM head.

## For Beginners

Mamba-2 is a faster version of the Mamba language model that
discovers a mathematical connection between state-space models and transformers. This
insight allows it to use optimized matrix multiplication hardware (like tensor cores on
GPUs) for a 2-8x speedup over the original Mamba, while maintaining the same constant
memory advantage during text generation.

## How It Works

Mamba-2 improves upon the original Mamba architecture by replacing the selective scan with a
structured state space duality (SSD) formulation that enables more efficient hardware utilization.

**Reference:** Dao and Gu, "Transformers are SSMs", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Mamba-2 blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

