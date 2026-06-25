---
title: "StableSRModel<T>"
description: "StableSR model for exploiting diffusion prior for real-world image super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

StableSR model for exploiting diffusion prior for real-world image super-resolution.

## For Beginners

StableSR uses Stable Diffusion's knowledge for image upscaling.

How StableSR works:

1. The degraded image is encoded to latent space by a time-aware encoder
2. The CFW module wraps features from the encoder and injects them into the frozen SD U-Net
3. The U-Net generates realistic details guided by these features
4. A controllable strength parameter balances fidelity vs quality
5. More diffusion steps (200 default) produce higher quality

Key characteristics:

- Leverages pretrained SD 1.5 prior for realistic detail generation
- Controllable fidelity-quality trade-off via CFW module
- Time-aware encoder adapts to diffusion timestep for better results
- Better perceptual quality than pure regression methods (PSNR-focused)
- 200 inference steps by default (more than standard diffusion models)

When to use StableSR:

- Upscaling photographs where perceptual quality matters more than pixel accuracy
- Restoring images where you want natural-looking details
- When you need fine control over fidelity-quality balance
- Combining with SD ecosystem tools (ControlNet, LoRA)

Limitations:

- Slower than GAN-based SR methods (200 inference steps)
- May generate details not present in the original image
- Requires more VRAM than lightweight SR models
- Best at 512x512 output (SD 1.5 native resolution)

## How It Works

StableSR leverages the generative prior of a pretrained Stable Diffusion model for
real-world image super-resolution. It introduces a controllable feature wrapping (CFW)
module and a time-aware encoder that adapts features to the diffusion timestep for
balancing fidelity to the input and perceptual quality.

Architecture components:

- Pretrained SD 1.5 U-Net backbone (frozen during SR training)
- Controllable Feature Wrapping (CFW) module for fidelity-quality balance
- Time-aware encoder that adapts to diffusion timestep
- Standard SD VAE (4-channel latent space, 0.18215 scale factor)
- CLIP ViT-L/14 text encoder (768-dim cross-attention)

Technical specifications:

- Architecture: SD 1.5 U-Net + CFW module + time-aware encoder
- U-Net backbone: 320 base channels, [1, 2, 4, 4] multipliers (frozen)
- Cross-attention dimension: 768 (CLIP ViT-L/14)
- CFW: Controllable feature wrapping for each U-Net decoder block
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Default inference steps: 200 (higher than standard for SR quality)
- Default guidance scale: 7.5

Reference: Wang et al., "Exploiting Diffusion Prior for Real-World Image Super-Resolution", IJCV 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableSRModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of StableSRModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` | Counts the flat-API parameter surface (predictor + VAE). |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image from a text prompt using StableSR defaults. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs diffusion-prior-based image super-resolution. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom or default configurations. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the SD 1.5 U-Net backbone (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-L/14 output (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for StableSR (7.5). |
| `DEFAULT_INFERENCE_STEPS` | Default number of inference steps for StableSR (200). |
| `DefaultHeight` | Default image height for StableSR output. |
| `DefaultWidth` | Default image width for StableSR output. |
| `LATENT_CHANNELS` | Number of latent channels in the SD VAE (4). |
| `_conditioner` | Optional CLIP text encoder conditioning module. |
| `_unet` | The U-Net noise predictor using the frozen SD 1.5 backbone. |
| `_vae` | The standard VAE for encoding/decoding between pixel and latent space. |

