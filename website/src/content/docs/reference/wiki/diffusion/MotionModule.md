---
title: "MotionModule<T>"
description: "AnimateDiff motion module for injecting temporal awareness into image diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

AnimateDiff motion module for injecting temporal awareness into image diffusion models.

## For Beginners

The Motion Module is a plug-in temporal attention layer that can be added to any image diffusion model to enable video generation. It learns temporal motion patterns that are applied on top of existing spatial features.

## How It Works

**References:**

- Paper: "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning" (Guo et al., 2023)

The motion module is a plug-in temporal attention block that can be inserted into any
image diffusion UNet to add video generation capability. It consists of:

- Temporal self-attention (across frames for each spatial position)
- Positional encoding (sinusoidal temporal embeddings)
- Feed-forward network (MLP for feature transformation)
- Zero-initialized output projection (for stable training from image models)

Key innovation: The zero-initialization ensures that inserting the motion module
into a pretrained image model initially produces identity output, allowing the
temporal parameters to be trained without disrupting the spatial generation quality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MotionModule(Int32,Int32,Int32,Int32,Int32)` | Initializes a new AnimateDiff motion module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `NumFrames` | Gets the number of frames. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Applies the motion module: temporal attention + FFN with residual connections. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

