---
title: "PrototypeAlignmentLayer<T>"
description: "A learned prototype-alignment layer per Sun et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A learned prototype-alignment layer per Sun et al. 2024 "TEST: Text Prototype Aligned
Embedding to Activate LLM's Ability for Time Series". Maintains a bank of
`numPrototypes` learnable embeddings of dimension `embedDim`. For each input
token, computes cosine similarity to every prototype, softmax-normalizes, then
aggregates prototypes by weight — producing an aligned representation that lives in
the same prototype subspace as a frozen LLM's text-token embeddings.

## For Beginners

Think of this as a "translator" that maps each input
patch to the closest matches in a small library of learned reference embeddings
(prototypes). The output is a weighted blend of those reference embeddings, so
downstream layers (often a frozen language model) see inputs in a vocabulary they
already understand.

## How It Works

Forward: input `[B, N, D]` → cosine similarity with prototypes
`[K, D]` → softmax over K → weights `[B, N, K]` → aggregate
prototypes → output `[B, N, D]`.

The prototype bank is trainable and initialized with small random values. During
training, the prototypes learn to represent the "grammar" of time-series patches in a
way that is compatible with a downstream frozen LLM.

**Reference:** Sun, C. et al., "TEST: Text Prototype Aligned Embedding to
Activate LLM's Ability for Time Series", ICLR 2024.
``.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypeAlignmentLayer(Int32,Int32)` | Initializes a new `PrototypeAlignmentLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Persists the constructor arguments so the deserializer can rebuild this layer at the same shape. |
| `GetParameters` |  |
| `ResetState` |  |
| `UpdateParameters()` |  |

