---
title: "RealBasicVSRSharp<T>"
description: "RealBasicVSR-Sharp: perceptually-optimized real-world video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

RealBasicVSR-Sharp: perceptually-optimized real-world video super-resolution.

## For Beginners

There are two ways to measure video quality: mathematical accuracy
(PSNR) and visual quality (how it looks to your eyes). The base RealBasicVSR maximizes
PSNR, while this "Sharp" variant maximizes visual quality. The Sharp version produces
images with crisper textures and more natural-looking details.

**Usage:**

## How It Works

RealBasicVSR-Sharp (Chan et al., CVPR 2022) is the perceptual variant of RealBasicVSR:

- Same pre-cleaning module + BasicVSR backbone as RealBasicVSR
- Trained with perceptual loss (VGG feature matching) for better texture recovery
- Adds GAN discriminator loss for sharper, more photo-realistic outputs
- Produces visually sharper results at the cost of slightly lower PSNR

This variant is preferred when visual quality matters more than pixel accuracy,
such as for display on screens or social media.

**Reference:** "Investigating Tradeoffs in Real-World Video Super-Resolution"
(Chan et al., CVPR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealBasicVSRSharp(NeuralNetworkArchitecture<>,RealBasicVSRSharpOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a RealBasicVSR-Sharp model in native training mode. |
| `RealBasicVSRSharp(NeuralNetworkArchitecture<>,String,RealBasicVSRSharpOptions)` | Creates a RealBasicVSR-Sharp model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

