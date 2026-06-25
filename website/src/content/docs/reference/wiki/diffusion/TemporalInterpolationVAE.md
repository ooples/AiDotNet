---
title: "TemporalInterpolationVAE<T>"
description: "Temporal interpolation VAE that generates intermediate frames in latent space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Temporal interpolation VAE that generates intermediate frames in latent space.

## For Beginners

The Temporal Interpolation VAE supports variable frame rate encoding and decoding. It can compress videos at different temporal resolutions, enabling efficient processing of both fast-action and slow-motion content.

## How It Works

**References:**

- Paper: "FILM: Frame Interpolation for Large Motion" (Reda et al., 2022)
- Paper: "Stable Video Diffusion" (Blattmann et al., 2023)

The Temporal Interpolation VAE extends standard video VAE with the ability to generate
intermediate frames between keyframes. This enables:

- Frame rate upsampling (e.g., 8fps to 24fps) in latent space
- Smoother video generation by interpolating between diffusion outputs
- Multi-scale temporal generation (coarse keyframes then fine interpolation)

Architecture:

- Inherits spatial encoding/decoding from TemporalVAE
- Adds temporal interpolation network that operates in latent space
- Interpolation uses bidirectional temporal attention and flow estimation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalInterpolationVAE(Int32,Int32,Int32,Int32,Double)` | Initializes a new Temporal Interpolation VAE. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `InterpolationFactor` | Gets the temporal interpolation factor. |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `Decode(Tensor<>)` |  |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeWithDistribution(Tensor<>)` |  |
| `GetParameters` |  |
| `InterpolateLatent(Tensor<>,Tensor<>)` | Interpolates between two latent frames to generate an intermediate frame. |
| `SetParameters(Vector<>)` |  |

