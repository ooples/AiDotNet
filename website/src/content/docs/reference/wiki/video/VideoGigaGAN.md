---
title: "VideoGigaGAN<T>"
description: "VideoGigaGAN: towards detail-rich video super-resolution with large-scale GAN."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

VideoGigaGAN: towards detail-rich video super-resolution with large-scale GAN.

## For Beginners

VideoGigaGAN is like a very talented speed-painter. While
diffusion models gradually "develop" a high-res image from noise (slow but versatile),
VideoGigaGAN directly "paints" detail in one stroke -- much faster. A special "detail
shuttle" ensures real details are preserved and fake ones aren't invented, while
anti-aliasing prevents the annoying flickering between frames.

**Usage:**

## How It Works

VideoGigaGAN (Xu et al., CVPR 2025) is the first large-scale GAN for video SR:

- GigaGAN backbone: upscaled StyleGAN architecture with 1B+ parameters, generating

exceptional spatial detail in a single forward pass (faster than diffusion models)

- Feature propagation with anti-aliasing: temporal feature propagation uses anti-aliased

flow warping to prevent temporal aliasing artifacts that cause flickering

- High-frequency shuttle: a dedicated parallel pathway that extracts and preserves

genuine high-frequency details (edges, textures, text) from the input, preventing the
generator from hallucinating false details while maintaining real ones

- Temporal discriminator: a 3D discriminator evaluates both per-frame quality and

temporal consistency, penalizing flickering and motion artifacts

- Supports up to 8x upscaling with rich perceptual details

**Note:** The full VideoGigaGAN architecture (GigaGAN backbone, high-frequency shuttle,
temporal discriminator, anti-aliased flow warping) is available through ONNX inference mode.
Native training mode uses a simplified baseline encoder-decoder for research and fine-tuning.

**Reference:** "VideoGigaGAN: Towards Detail-rich Video Super-Resolution"
(Xu et al., CVPR 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoGigaGAN(NeuralNetworkArchitecture<>,String,VideoGigaGANOptions)` | Creates a VideoGigaGAN model in ONNX inference mode. |
| `VideoGigaGAN(NeuralNetworkArchitecture<>,VideoGigaGANOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a VideoGigaGAN model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

