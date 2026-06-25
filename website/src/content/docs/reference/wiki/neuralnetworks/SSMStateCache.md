---
title: "SSMStateCache<T>"
description: "Manages hidden state caching across autoregressive inference steps for SSM models."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Manages hidden state caching across autoregressive inference steps for SSM models.

## For Beginners

When a language model generates text one word at a time, it needs
to "remember" what it processed before. Instead of re-reading the entire text for each new word,
the state cache saves the model's memory from previous words.

Think of it like taking notes while reading:

- Without cache: re-read the entire book for each new sentence (slow!)
- With cache: just read your notes and the new sentence (fast!)

For SSM models, this cache stores:

- The "hidden state" (the model's current memory/summary)
- The "conv buffer" (recent context needed by the convolution layer)

## How It Works

During autoregressive generation (e.g., language model text generation), each new token
depends on the hidden states produced by all previous tokens. Rather than reprocessing
the entire sequence for each new token, the state cache stores the SSM hidden states
so only the new token needs to be processed. This is the SSM equivalent of the KV-cache
used in Transformer models.

The cache stores two types of state per layer:

1. SSM hidden state: the state space model's recurrent state [batch, innerDim, stateDim]
2. Conv buffer: the convolution buffer for Conv1D layers [batch, innerDim, convKernelSize-1]

This matches the state needed by Mamba's selective scan and Conv1D operations.

**Usage example:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSMStateCache(Boolean,Int32)` | Creates a new SSM state cache. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CachedLayerCount` | Gets the number of layers that have cached SSM states. |
| `CompressionEnabled` | Gets whether state precision reduction is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheConvBuffer(Int32,Tensor<>)` | Caches the Conv1D buffer state for a specific layer. |
| `CacheSSMState(Int32,Tensor<>)` | Caches the SSM hidden state for a specific layer. |
| `Clone` | Creates a deep copy of this cache, useful for beam search where multiple hypotheses need independent state copies. |
| `GetConvBuffer(Int32)` | Retrieves the cached Conv1D buffer for a specific layer. |
| `GetConvBufferLayerIndices` | Gets all layer indices that have cached conv buffers. |
| `GetMemoryUsageBytes` | Gets the approximate memory usage of the cache in bytes. |
| `GetSSMState(Int32)` | Retrieves the cached SSM hidden state for a specific layer. |
| `GetSSMStateLayerIndices` | Gets all layer indices that have cached SSM states. |
| `HasConvBuffer(Int32)` | Checks whether a specific layer has a cached conv buffer. |
| `HasSSMState(Int32)` | Checks whether a specific layer has a cached SSM state. |
| `Reset` | Clears all cached states, preparing for a new sequence. |

