---
title: "ImprovedVideoVAE<T>"
description: "Improved Video VAE with temporal-aware compression and motion consistency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Improved Video VAE with temporal-aware compression and motion consistency.

## For Beginners

A video is like a stack of images. This VAE compresses both
spatially (each frame gets smaller) and temporally (every 4 frames become 1 latent frame).
It understands motion — if a ball is moving right, the latent representation captures
that smoothly rather than treating each frame independently.

## How It Works

Extends standard image VAE with temporal convolutions and motion-aware encoding.
Uses causal temporal convolutions (only looking at past frames) for streaming-compatible
encoding, and temporal attention for capturing long-range motion patterns. Achieves
4x temporal compression on top of 8x spatial compression.

Reference: Improved upon CogVideoX VAE architecture with motion-aware encoding, 2024-2025

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImprovedVideoVAE(Int32,Int32,Int32,Int32,ILossFunction<>,Nullable<Int32>)` | Initializes a new Improved Video VAE. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsSlicing` |  |
| `SupportsTiling` |  |
| `TemporalDownsampleFactor` | Gets the temporal downsampling factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `Decode(Tensor<>)` |  |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeVideo(List<Tensor<>>)` | Encodes a video (sequence of frames) with temporal compression. |
| `EncodeWithDistribution(Tensor<>)` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

