---
title: "RewardGuidance<T>"
description: "Reward-guided sampling for inference-time alignment of diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Alignment`

Reward-guided sampling for inference-time alignment of diffusion models.

## For Beginners

Instead of permanently changing the model (like RLHF does),
reward guidance acts like a compass during image generation. At each denoising step,
it asks the reward model "which direction leads to better images?" and nudges the
generation that way. This is flexible — you can change the reward model or guidance
strength without retraining.

## How It Works

Reward guidance modifies the sampling process at inference time by incorporating
gradients from a reward model. Unlike RLHF which fine-tunes the model, reward
guidance keeps the base model frozen and steers the denoising trajectory toward
higher-reward regions using the reward model's gradient signal.

Reference: Xu et al., "Imagereward: Learning and evaluating human preferences for text-to-image generation", NeurIPS 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RewardGuidance(Double,Double,Double)` | Initializes a new reward guidance module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GradientClipNorm` | Gets the gradient clipping norm. |
| `GuidanceScale` | Gets the guidance scale. |
| `TruncationTimestep` | Gets the truncation timestep fraction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGuidance(Vector<>,Vector<>,Double)` | Applies reward guidance to a noise prediction by incorporating the reward gradient. |
| `ClipGradientNorm(Vector<>)` | Clips the gradient vector to have maximum L2 norm. |

