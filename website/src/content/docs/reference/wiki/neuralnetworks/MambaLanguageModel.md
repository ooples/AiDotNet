---
title: "MambaLanguageModel<T>"
description: "Implements a full Mamba language model: token embedding + N MambaBlocks + RMS normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full Mamba language model: token embedding + N MambaBlocks + RMS normalization + LM head.

## For Beginners

Mamba is an efficient alternative to Transformers that processes
sequences in linear time instead of quadratic time, making it much faster for long sequences
while maintaining competitive quality.

## How It Works

This assembles the complete Mamba architecture as described in the original paper.
Mamba uses selective state spaces with input-dependent gating to achieve linear-time
sequence modeling with competitive quality to Transformers.

**Reference:** Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Mamba blocks. |
| `StateDimension` | Gets the SSM state dimension. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateStepState(Int32)` | Creates fresh decoding state for incremental, O(n) token-by-token generation. |
| `GetOptions` |  |
| `Step(Tensor<>,MambaModelState<>)` | Processes a single token using carried KV-cache state, returning the same logits the full-sequence `Predict` would produce at that position — but in O(1) work per token rather than reprocessing the prefix. |

