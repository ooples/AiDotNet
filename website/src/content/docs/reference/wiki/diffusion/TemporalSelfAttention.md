---
title: "TemporalSelfAttention<T>"
description: "Temporal self-attention layer for video diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Temporal self-attention layer for video diffusion models.

## For Beginners

Temporal Self-Attention computes attention exclusively along the time dimension - each spatial position attends to the same position across all frames. This captures how individual pixels or regions change over time.

## How It Works

**References:**

- Paper: "Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models" (Blattmann et al., 2023)
- Paper: "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models" (Guo et al., 2023)

Temporal self-attention applies attention across the time dimension of video features.
For each spatial position, tokens from all frames attend to each other, enabling the model
to learn temporal relationships and maintain consistency across frames.

Architecture:

- Input: [batch * height * width, frames, channels]
- Reshapes spatial dims into batch for per-position temporal attention
- Each spatial location independently attends across frames
- Output: same shape as input with temporal information mixed

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalSelfAttention(Int32,Int32,Int32,Int32)` | Initializes a new temporal self-attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `NumFrames` | Gets the number of frames. |
| `NumHeads` | Gets the number of attention heads. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs temporal self-attention across frames for each spatial position. |
| `GetDiagnostics` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

