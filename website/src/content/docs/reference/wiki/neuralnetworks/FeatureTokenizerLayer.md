---
title: "FeatureTokenizerLayer<T>"
description: "Feature tokenizer for tabular transformers: embeds each scalar input feature into its OWN learnable embedding vector, producing a `[features, embedding]` token sequence that a transformer encoder can attend over."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Feature tokenizer for tabular transformers: embeds each scalar input feature into its OWN
learnable embedding vector, producing a `[features, embedding]` token sequence that a
transformer encoder can attend over.

## How It Works

Implements the numerical feature tokenizer of FT-Transformer (Gorishniy et al. 2021,
"Revisiting Deep Learning Models for Tabular Data"), the same per-column embedding idea that
underlies TabTransformer (Huang et al. 2020): token[f] = x[f] · W[f] + b[f], where each feature
f has its own embedding row W[f] (shape `[embedding]`) and bias b[f].

This is critical for tabular models: a shared projection (a single Dense layer) maps the whole
feature vector to ONE vector, so self-attention runs over a length-1 sequence (no attention) —
and even a per-token Dense produces collinear tokens (all `x_f · W`) whose only difference
is a scalar that LayerNorm then removes, collapsing distinct inputs to identical outputs.
Per-feature embedding directions break that degeneracy and encode feature identity, so no
separate positional embedding is needed.

The feature count is resolved lazily from the first forward input (like `DenseLayer`),
so the layer adapts to the actual fed input width even when a model's declared input size differs.
Output is always a batched `[batch, features, embedding]` tensor (batch=1 for an unbatched
`[features]` input) so the downstream encoder and head treat the feature axis unambiguously.
Forward is expressed with broadcast Engine ops on the registered weight/bias tensors so the tape
computes their gradients automatically.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureTokenizerLayer(Int32)` | Initializes a tokenizer whose feature count is resolved lazily on the first forward pass. |
| `FeatureTokenizerLayer(Int32,Int32)` | Initializes a tokenizer with an explicit feature count. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

