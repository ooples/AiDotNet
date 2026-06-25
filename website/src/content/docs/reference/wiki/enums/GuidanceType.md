---
title: "GuidanceType"
description: "Types of guidance methods for diffusion model inference."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of guidance methods for diffusion model inference.

## For Beginners

Guidance is like giving the AI artist instructions on how
closely to follow your prompt. Different methods trade off between creativity
and accuracy in different ways.

## How It Works

Guidance methods control how the diffusion model balances quality, diversity,
and adherence to conditioning signals during generation.

## Fields

| Field | Summary |
|:-----|:--------|
| `AdaptiveProjected` | Adaptive Projected Guidance (APG) — projects guidance to reduce artifacts. |
| `ClassifierFree` | Standard Classifier-Free Guidance (CFG) using conditional/unconditional interpolation. |
| `DynamicCFG` | Dynamic CFG — adjusts guidance scale per timestep for better quality. |
| `ELLA` | ELLA (Efficient Large Language Model Adapter) guidance for enhanced text understanding. |
| `None` | No guidance — unconditional generation. |
| `PerturbedAttention` | Perturbed Attention Guidance (PAG) — uses attention perturbation instead of negative prompt. |
| `RescaledCFG` | Rescaled CFG — rescales the guided prediction to prevent over-saturation. |
| `SelfAttention` | Self-Attention Guidance (SAG) — leverages self-attention maps for adaptive guidance. |

