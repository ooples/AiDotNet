---
title: "MARModel<T>"
description: "Masked Autoregressive (MAR) model for image generation via masked token prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Masked Autoregressive (MAR) model for image generation via masked token prediction.

## For Beginners

MAR treats image generation like filling in a puzzle. It starts
with all pieces hidden (masked) and progressively reveals them, predicting what each
piece looks like based on the already-revealed pieces. This approach bridges
autoregressive models (like GPT) and diffusion models.

## How It Works

MAR generates images by progressively predicting masked tokens in a discrete latent
space. Unlike standard diffusion which adds/removes continuous noise, MAR works with
discrete tokens and uses a masking strategy that enables flexible generation ordering
and variable-speed generation.

Reference: Li et al., "Autoregressive Image Generation without Vector Quantization", NeurIPS 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

