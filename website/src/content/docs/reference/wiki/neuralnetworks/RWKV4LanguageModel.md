---
title: "RWKV4LanguageModel<T>"
description: "Implements a full RWKV-4 language model: token embedding + N RWKVLayer blocks + layer normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full RWKV-4 language model: token embedding + N RWKVLayer blocks + layer normalization + LM head.

## For Beginners

RWKV-4 is a language model that generates text like GPT, but runs
much faster because it processes text in linear time instead of quadratic time.

How it works:

1. Each word is converted to a vector (embedding)
2. Multiple RWKV layers process the vectors, building understanding of context
3. The output is probabilities for what the next word should be

What makes it special:

- Processes text in linear time (twice as long text takes twice as long, not four times)
- Uses constant memory per token during generation
- First RWKV version to achieve competitive quality with Transformers

Real-world examples: RWKV-4 models from 169M to 14B parameters.

## How It Works

This assembles the complete RWKV-4 architecture as described in the original paper:

RWKV-4 is the first widely-adopted version of the RWKV architecture. It replaces standard
multi-head attention with a linear-complexity WKV (Weighted Key Value) mechanism that uses
fixed exponential decay to weight past tokens. The channel mixing sub-layer replaces the
standard FFN with a squared ReLU gating mechanism.

Key characteristics of RWKV-4:

- Fixed learned time decay per channel (not data-dependent)
- Single-head WKV attention (no matrix-valued states)
- Squared ReLU channel mixing: sigmoid(r) * (W_v * max(k, 0)^2)
- Token shift mixing with fixed learned coefficients
- O(n) time complexity and O(1) memory per token during generation

**Reference:** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023.
https://arxiv.org/abs/2305.13048

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RWKV4LanguageModel(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,ILossFunction<>,RWKV4Options)` | Creates an RWKV-4 language model using native library layers. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of RWKV-4 blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `UpdateParameters(Vector<>)` |  |

