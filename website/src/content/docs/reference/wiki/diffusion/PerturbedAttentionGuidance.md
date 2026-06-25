---
title: "PerturbedAttentionGuidance<T>"
description: "Perturbed Attention Guidance (PAG) for diffusion model inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Guidance`

Perturbed Attention Guidance (PAG) for diffusion model inference.

## For Beginners

Instead of comparing "with prompt" vs "without prompt",
PAG compares "normal attention" vs "broken attention." This gives better image
quality, especially at high guidance scales where CFG can cause artifacts.

## How It Works

PAG replaces the unconditional prediction with a "perturbed" prediction where
self-attention maps are modified (e.g., replaced with identity). This produces
better guidance than standard CFG without needing a separate unconditional pass.

Reference: Ahn et al., "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance", ECCV 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerturbedAttentionGuidance(Double)` | Initializes a new Perturbed Attention Guidance instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` |  |

