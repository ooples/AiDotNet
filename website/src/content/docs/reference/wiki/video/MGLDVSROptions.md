---
title: "MGLDVSROptions"
description: "Configuration options for the MGLD-VSR motion-guided latent diffusion video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the MGLD-VSR motion-guided latent diffusion video super-resolution model.

## For Beginners

While most diffusion-based upscalers treat each frame independently
(leading to flickering), MGLD-VSR explicitly tells the AI "here is how objects moved"
using optical flow, so it can maintain smooth, consistent motion across frames.

## How It Works

MGLD-VSR (Yang et al., 2024) integrates explicit motion guidance into latent diffusion:

- Motion-guided denoising: optical flow maps are injected as additional conditioning

into each denoising step, ensuring temporal consistency

- Latent diffusion: operates in a compressed latent space (VAE encoder/decoder) for

efficiency, with the U-Net denoiser conditioned on both the LR frames and flow

- Motion-aware loss: penalizes temporal inconsistency in addition to pixel accuracy

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MGLDVSROptions` | Initializes a new instance with default values. |
| `MGLDVSROptions(MGLDVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LatentDim` | Gets or sets the latent space dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `MotionGuidanceWeight` | Gets or sets the motion guidance weight. |
| `NumDenoisingSteps` | Gets or sets the number of denoising steps. |
| `NumFeatures` | Gets or sets the number of UNet feature channels. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the U-Net. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |

