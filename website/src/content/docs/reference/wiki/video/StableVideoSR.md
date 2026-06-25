---
title: "StableVideoSR<T>"
description: "StableVideoSR: stable diffusion with temporal conditioning for video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

StableVideoSR: stable diffusion with temporal conditioning for video super-resolution.

## For Beginners

StableVideoSR extends the popular Stable Diffusion image AI to
handle video. It adds "temporal awareness" so the model considers what happened in
previous and next frames when upscaling each frame, preventing the flickering that
occurs when frames are processed independently.

**Usage:**

## How It Works

StableVideoSR (2024) adapts the Stable Diffusion architecture for video SR:

- Temporal conditioning modules: cross-attention layers inserted between spatial attention

in the U-Net attend to features from adjacent frames, maintaining temporal coherence

- ControlNet adapter: a trainable copy of the U-Net encoder provides fine-grained spatial

conditioning from the low-resolution input during the denoising process

- Classifier-free guidance: balances between faithful reconstruction and generative

enhancement during inference

- Noise schedule: adapted from image diffusion to preserve temporal structure

**Reference:** "StableVideoSR: Video Super-Resolution via Stable Diffusion with
Temporal Conditioning" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableVideoSR(NeuralNetworkArchitecture<>,StableVideoSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a StableVideoSR model in native training mode. |
| `StableVideoSR(NeuralNetworkArchitecture<>,String,StableVideoSROptions)` | Creates a StableVideoSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

