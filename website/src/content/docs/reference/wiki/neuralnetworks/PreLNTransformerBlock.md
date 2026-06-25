---
title: "PreLNTransformerBlock<T>"
description: "Pre-Layer-Normalization transformer block with RMSNorm and a caller-supplied self-attention sublayer."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Pre-Layer-Normalization transformer block with RMSNorm and a caller-supplied
self-attention sublayer. Matches the decoder-style architecture used by T5,
LLaMA, Gemma, Qwen2, and ChatGLM3 text encoders / language models.

## How It Works

Block structure (Raffel 2020 §2.1, Touvron 2023 §2.2):

The attention sublayer is injected through the constructor so the same
block class supports T5's relative-bias attention, Gemma's RoPE
multi-head attention, Qwen2's RoPE grouped-query attention, and
ChatGLM3's RoPE multi-query attention. The FFN is a paper-canonical
two-matrix linear → activation → linear stack with no biases (matching
the T5 / LLaMA / Gemma / Qwen2 / ChatGLM3 convention).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PreLNTransformerBlock(Int32,Int32,LayerBase<>,IActivationFunction<>)` | Initialises a pre-LN transformer block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` | Forward pass. |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

