---
title: "SeedVROptions"
description: "Configuration options for the SeedVR diffusion transformer video restoration model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the SeedVR diffusion transformer video restoration model.

## For Beginners

SeedVR uses a powerful video generation AI model and adapts it
to fix degraded videos. It works like a smart upscaler that can handle many types
of damage (noise, blur, compression) because it learned from millions of clean videos
what natural video should look like.

## How It Works

SeedVR (Wang et al., 2025) uses a diffusion transformer (DiT) architecture for generic
video restoration (SR, denoising, deblurring, JPEG artifact removal):

- Shifted window attention: efficient spatio-temporal self-attention with window shifting

for cross-window interaction (similar to Swin Transformer but in 3D)

- Causal temporal attention: maintains temporal consistency without future frame access
- Progressive upsampling: multi-stage 2x upsampling from latent to pixel space
- Text-to-video diffusion priors: initialized from a pretrained T2V model

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeedVROptions` | Initializes a new instance with default values. |
| `SeedVROptions(SeedVROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDenoisingSteps` | Gets or sets the number of denoising steps. |
| `NumDiTBlocks` | Gets or sets the number of DiT transformer blocks. |
| `NumFeatures` | Gets or sets the number of feature channels in the DiT blocks. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PatchSize` | Gets or sets the patch size for tokenizing video frames. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |
| `WindowSize` | Gets or sets the local window size for shifted window attention. |

