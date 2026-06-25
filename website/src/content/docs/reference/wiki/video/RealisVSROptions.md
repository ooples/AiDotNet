---
title: "RealisVSROptions"
description: "Configuration options for the RealisVSR detail-enhanced diffusion 4K video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the RealisVSR detail-enhanced diffusion 4K video super-resolution model.

## For Beginners

RealisVSR uses an AI video generation model (Wan 2.1) to upscale
real-world video to 4K. It adds a special "detail enhancement" module that makes sure
fine details like text and textures stay sharp during the upscaling process, while the
video generation backbone ensures smooth, natural-looking motion.

## How It Works

RealisVSR (2025) achieves coherent 4K real-world video super-resolution through:

- Wan 2.1 video diffusion backbone with detail-enhancement ControlNet adapter
- Motion-aware temporal conditioning for consistent inter-frame motion
- Detail-enhancement module that preserves fine textures during the diffusion process
- Designed specifically for upscaling real-world degraded video to 4K resolution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealisVSROptions` | Initializes a new instance with default values. |
| `RealisVSROptions(RealisVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ControlNetScale` | Gets or sets the ControlNet conditioning scale for detail enhancement. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDenoisingSteps` | Gets or sets the number of denoising steps. |
| `NumFeatures` | Gets or sets the number of UNet feature channels. |
| `NumResBlocks` | Gets or sets the number of residual blocks per UNet level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |

