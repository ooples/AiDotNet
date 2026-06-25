---
title: "StableVideoSROptions"
description: "Configuration options for the StableVideoSR temporal-conditioned diffusion video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the StableVideoSR temporal-conditioned diffusion video super-resolution model.

## For Beginners

StableVideoSR takes the popular Stable Diffusion image AI and
extends it to handle video. It adds special "temporal" modules that look at neighboring
frames to ensure the output video is smooth and flicker-free, not just a sequence
of independently upscaled images.

## How It Works

StableVideoSR (2024) adapts the Stable Diffusion architecture for video SR:

- Temporal conditioning modules: inserted between spatial attention layers in the U-Net,

these cross-attend to features from adjacent frames for temporal coherence

- ControlNet adapter: a frozen copy of the encoder provides fine-grained spatial control

from the low-resolution input while the main U-Net generates high-resolution output

- Classifier-free guidance: balances restoration fidelity vs generative quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableVideoSROptions` | Initializes a new instance with default values. |
| `StableVideoSROptions(StableVideoSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ControlNetScale` | Gets or sets the ControlNet conditioning scale. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LatentDim` | Gets or sets the latent space dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDenoisingSteps` | Gets or sets the number of denoising steps. |
| `NumFeatures` | Gets or sets the number of UNet feature channels. |
| `NumTemporalLayers` | Gets or sets the number of temporal attention layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |

