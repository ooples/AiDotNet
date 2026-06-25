---
title: "ControlARModel<T>"
description: "ControlAR model combining autoregressive generation with spatial control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlAR model combining autoregressive generation with spatial control.

## For Beginners

While standard ControlNet works with diffusion models,
ControlAR brings the same "follow my reference image" capability to autoregressive
(token-based) image generators, bridging the two main approaches to image generation.

## How It Works

ControlAR adapts ControlNet-style spatial conditioning for autoregressive image
generation models. It enables token-level control where spatial conditions are
mapped to discrete token sequences for AR model consumption.

Reference: Li et al., "ControlAR: Controllable Image Generation with Autoregressive Models", 2024

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

