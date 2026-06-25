---
title: "RealBasicVSRSharpOptions"
description: "Configuration options for the RealBasicVSR-Sharp perceptually-optimized video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the RealBasicVSR-Sharp perceptually-optimized video super-resolution model.

## For Beginners

The "Sharp" variant trades mathematical accuracy for visual quality.
While the base RealBasicVSR optimizes for pixel-perfect reconstruction (high PSNR),
this variant uses perceptual and adversarial losses to produce results that look sharper
and more natural to human eyes, even if individual pixels differ slightly.

## How It Works

RealBasicVSR-Sharp (Chan et al., CVPR 2022) is the perceptual variant of RealBasicVSR:

- Uses perceptual loss (VGG feature matching) instead of pixel-only L1 loss
- Adds GAN discriminator loss for sharper, more realistic textures
- Same pre-cleaning module and BasicVSR backbone as the base variant
- Produces visually sharper results at the cost of slightly lower PSNR

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealBasicVSRSharpOptions` | Initializes a new instance with default values. |
| `RealBasicVSRSharpOptions(RealBasicVSRSharpOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CleaningModuleBlocks` | Gets or sets the number of residual blocks in the pre-cleaning module. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `GANWeight` | Gets or sets the GAN discriminator loss weight. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFrames` | Gets or sets the number of input frames. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the BasicVSR backbone. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PerceptualWeight` | Gets or sets the perceptual loss weight. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `WarmupSteps` | Gets or sets the warmup steps. |

