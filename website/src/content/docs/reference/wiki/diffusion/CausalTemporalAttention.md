---
title: "CausalTemporalAttention<T>"
description: "Causal temporal attention for autoregressive video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Causal temporal attention for autoregressive video generation.

## For Beginners

Causal Temporal Attention only looks at past and current frames (never future frames) when processing video. This is essential for streaming and autoregressive generation where future frames have not been generated yet.

## How It Works

**References:**

- Paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (2024)
- Paper: "Cosmos World Foundation Model" (NVIDIA, 2024)

Causal temporal attention ensures that each frame can only attend to previous frames
and itself, not future frames. This is essential for:

- Streaming/real-time video generation
- Autoregressive frame-by-frame generation
- World models where future depends only on past

Architecture:

- Uses a causal mask to prevent attending to future frames
- Frame t can attend to frames 0, 1, ..., t but not t+1, t+2, ...
- Combined with spatial attention for full spatio-temporal modeling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalTemporalAttention(Int32,Int32,Int32)` | Initializes a new causal temporal attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `IsCausal` | Whether causal masking is enabled. |
| `NumFrames` | Gets the number of frames. |
| `NumHeads` | Gets the number of attention heads. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs causal temporal attention where each frame attends only to past frames. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

