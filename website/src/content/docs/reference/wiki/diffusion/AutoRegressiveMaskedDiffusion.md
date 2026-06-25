---
title: "AutoRegressiveMaskedDiffusion<T>"
description: "Autoregressive Masked Diffusion for hybrid discrete-continuous image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Autoregressive Masked Diffusion for hybrid discrete-continuous image generation.

## For Beginners

This model works in two interleaved phases: first it decides
which parts of the image to reveal (like solving a jigsaw puzzle), then it refines
those revealed parts using diffusion (smoothing out the details). This two-phase
approach naturally builds images from rough structure to fine detail.

## How It Works

Combines masked token prediction with diffusion denoising in a unified framework.
The model alternates between predicting which tokens to unmask (discrete) and
refining the unmasked token values through diffusion (continuous), creating a
natural curriculum from coarse structure to fine details.

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

