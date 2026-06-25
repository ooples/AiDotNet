---
title: "Full3DAttention<T>"
description: "Full 3D attention across all spatio-temporal positions simultaneously."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Full 3D attention across all spatio-temporal positions simultaneously.

## For Beginners

Full 3D Attention jointly attends to all spatial and temporal positions simultaneously. While computationally expensive, it captures the richest interactions between space and time for highest quality results.

## How It Works

**References:**

- Paper: "Sora: Creating video from text" (OpenAI, 2024)
- Paper: "MovieGen: A Cast of Media Foundation Models" (Meta, 2024)

Full 3D attention computes attention across all tokens in the video volume simultaneously
(all frames, all spatial positions). While computationally expensive at O((T*H*W)^2),
it captures the richest spatio-temporal interactions and is used in the most capable
video generation models like Sora and MovieGen.

Architecture:

- Flattens all frames and spatial positions into a single sequence
- Applies multi-head self-attention across the entire video volume
- Uses Flash Attention for memory efficiency
- Uses Flash Attention implementation for O(N) memory

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Full3DAttention(Int32,Int32,Int32,Int32)` | Initializes a new full 3D attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `SupportsTraining` |  |
| `TotalSequenceLength` | Gets the total sequence length (frames * height * width). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Applies full 3D attention across all spatio-temporal positions. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

