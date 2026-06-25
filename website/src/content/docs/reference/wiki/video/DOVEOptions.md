---
title: "DOVEOptions"
description: "Configuration options for the DOVE video diffusion prior restoration model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the DOVE video diffusion prior restoration model.

## For Beginners

DOVE uses a large video generation AI model (similar to those
that create videos from text) but repurposes it for restoration. Instead of generating
new content, it generates clean versions of degraded videos by leveraging the model's
knowledge of what natural video looks like.

## How It Works

DOVE (Chen et al., 2025) harnesses large-scale video diffusion models as priors:

- Degradation estimation stage: analyzes the input to determine degradation type/level
- Guided generation stage: conditions a pretrained video diffusion model (SVD backbone)

on the degradation estimate for restoration-aware denoising

- Temporal consistency: video diffusion priors inherently maintain frame coherence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DOVEOptions` | Initializes a new instance with default values. |
| `DOVEOptions(DOVEOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LatentDim` | Gets or sets the latent space dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in the UNet. |
| `NumDenoisingSteps` | Gets or sets the number of denoising steps in the diffusion process. |
| `NumFeatures` | Gets or sets the number of UNet feature channels. |
| `NumResBlocks` | Gets or sets the number of residual blocks per UNet level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |

