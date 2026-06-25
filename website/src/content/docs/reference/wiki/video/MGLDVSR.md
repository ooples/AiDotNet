---
title: "MGLDVSR<T>"
description: "MGLD-VSR: motion-guided latent diffusion for video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

MGLD-VSR: motion-guided latent diffusion for video super-resolution.

## For Beginners

Diffusion models generate images by gradually removing noise.
MGLD-VSR tells the noise-removal process exactly how objects are moving between frames
(using optical flow), so it can produce temporally consistent video rather than
independently generated frames that might flicker.

**Usage:**

## How It Works

MGLD-VSR (Yang et al., 2024) integrates explicit motion guidance into latent diffusion:

- Motion-guided denoising: estimated optical flow maps are injected as conditioning signals

at each denoising step, explicitly teaching the model about temporal relationships

- Latent diffusion: operates in a compressed latent space via a VAE encoder/decoder,

making the diffusion process computationally efficient for video

- Motion-aware loss: combines pixel-level L2 loss with a temporal warping consistency term

that penalizes artifacts at motion boundaries

The key innovation is making the diffusion process motion-aware, rather than relying
solely on the model to implicitly learn temporal consistency from data.

**Reference:** "MGLD-VSR: Motion-Guided Latent Diffusion for Video Super-Resolution"
(Yang et al., 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MGLDVSR(NeuralNetworkArchitecture<>,MGLDVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MGLD-VSR model in native training mode. |
| `MGLDVSR(NeuralNetworkArchitecture<>,String,MGLDVSROptions)` | Creates a MGLD-VSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

