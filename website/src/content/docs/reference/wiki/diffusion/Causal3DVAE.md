---
title: "Causal3DVAE<T>"
description: "Causal 3D VAE for video with temporal causal convolutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Causal 3D VAE for video with temporal causal convolutions.

## For Beginners

The Causal 3D VAE compresses video into a much smaller latent space while preserving temporal information. Causal means it only uses past frames to encode each frame, enabling streaming video compression.

## How It Works

**References:**

- Paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (2024)
- Paper: "Open-Sora Plan" (2024)

The Causal 3D VAE uses causal 3D convolutions to encode and decode video. Causal convolutions
ensure that each frame's encoding depends only on the current and previous frames, enabling:

- Streaming video generation (encode/decode frame by frame)
- Autoregressive generation without future frame leakage
- Temporal compression (e.g., 4x in time dimension)

Architecture:

- Encoder: Causal 3D Conv blocks with temporal stride for temporal compression
- Decoder: Causal 3D TransposeConv blocks for temporal upsampling
- Both spatial and temporal compression in latent space
- Typical compression: 8x spatial, 4x temporal

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Causal3DVAE(Int32,Int32,Int32,Int32[],Int32,Double)` | Initializes a new Causal 3D VAE. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `TemporalCompression` | Gets the temporal compression factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `Decode(Tensor<>)` |  |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeWithDistribution(Tensor<>)` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

