---
title: "VideoGigaGANOptions"
description: "Configuration options for the VideoGigaGAN large-scale GAN for video SR."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the VideoGigaGAN large-scale GAN for video SR.

## For Beginners

VideoGigaGAN is like having a very talented artist with a
magnifying glass. While diffusion models (like StableVideoSR) gradually "develop"
a high-res image from noise, VideoGigaGAN directly "paints" detail in a single
forward pass -- much faster. A special "detail shuttle" ensures real details are
preserved while fake ones are avoided, and anti-aliasing prevents flickering.

## How It Works

VideoGigaGAN (Xu et al., CVPR 2025) is the first large-scale GAN for video SR:

- GigaGAN backbone: upscaled StyleGAN architecture with 1B+ parameters, providing

exceptional detail generation capability beyond diffusion models in sharpness

- Feature propagation with anti-aliasing: temporal feature propagation using

anti-aliased flow warping to prevent temporal aliasing artifacts

- High-frequency shuttle: a dedicated pathway that extracts and preserves high-frequency

details (edges, textures) from the input through a parallel processing stream,
preventing the GAN from hallucinating false details

- Temporal discriminator: a 3D discriminator that evaluates temporal consistency

in addition to per-frame quality

- Supports up to 8x upscaling with rich perceptual details

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoGigaGANOptions` | Initializes a new instance with default values. |
| `VideoGigaGANOptions(VideoGigaGANOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `GANWeight` | Gets or sets the weight for the GAN adversarial loss component. |
| `HFShuttleWeight` | Gets or sets the weight for the high-frequency shuttle loss. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels in the generator backbone. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the generator. |
| `NumStyleLayers` | Gets or sets the number of style mixing layers in the GigaGAN generator. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PerceptualWeight` | Gets or sets the weight for the perceptual (LPIPS) loss component. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |

