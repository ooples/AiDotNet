---
title: "PositionalEncodingLayer<T>"
description: "Represents a layer that adds positional encodings to input sequences."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that adds positional encodings to input sequences.

## For Beginners

This layer adds information about position to your sequence data.

Think of it like numbering the words in a sentence:

- Without position information, a model only knows which words are in the sentence
- With position information, it knows which word comes first, second, third, etc.

For example, the sentences "dog bites man" and "man bites dog" contain the same words
but have completely different meanings because of word order. Positional encoding
helps models understand this difference.

The layer uses a clever mathematical pattern of sine and cosine waves to encode positions.
This approach has several advantages:

- It creates a unique pattern for each position
- Similar positions have similar encodings (helpful for generalization)
- It can potentially handle sequences longer than those seen during training
- The encodings have consistent patterns that models can learn from

## How It Works

The PositionalEncodingLayer adds position-dependent signals to input embeddings, which helps
sequence models like Transformers understand the order of elements in a sequence. Since
attention-based models have no inherent notion of sequence order, positional encodings
provide this critical information. The encodings use sine and cosine functions of different
frequencies to create unique position-dependent patterns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PositionalEncodingLayer(Int32,Int32)` | Initializes a new instance of the `PositionalEncodingLayer` class with the specified maximum sequence length and embedding size. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets whether this layer supports GPU execution. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the positional encoding layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU, adding positional encodings to input embeddings. |
| `GetParameters` | Gets all trainable parameters from the positional encoding layer as a single vector. |
| `InitializeEncodings` | Initializes the positional encodings using sine and cosine functions. |
| `ResetState` | Resets the internal state of the positional encoding layer. |
| `UpdateParameters()` | Updates the parameters of the positional encoding layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `embeddingSize` | The size of each embedding vector. |
| `encodings` | The pre-computed positional encodings tensor. |
| `maxSequenceLength` | The maximum sequence length that this layer can handle. |

