---
title: "SDUpscalerModel<T>"
description: "Stable Diffusion x4 Upscaler model for text-guided latent super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

Stable Diffusion x4 Upscaler model for text-guided latent super-resolution.

## For Beginners

This model increases image resolution by 4x with text guidance.

How the SD Upscaler works:

1. The low-resolution image is resized and concatenated with latent noise (7 input channels)
2. A text prompt guides what details to add during upscaling
3. The U-Net denoises in latent space over 75 steps
4. The VAE decodes the result to a 4x larger image

Key characteristics:

- 4x upscaling (128 to 512, 256 to 1024, etc.)
- Text-guided: prompts control what details are added
- 7 input channels: 4 latent + 3 low-res RGB conditioning
- Based on SD 2.x architecture with OpenCLIP text encoder
- DDIM scheduler for efficient 75-step inference

When to use SD Upscaler:

- Upscaling images with specific desired detail types
- Combining upscaling with style transfer
- When you want prompt control over the upscaling process
- AI-generated image enhancement

Limitations:

- Fixed 4x upscale factor
- Slower than Real-ESRGAN due to 75 inference steps
- May add details inconsistent with original content at high guidance

## How It Works

The Stable Diffusion x4 Upscaler takes a low-resolution image and upscales it 4x
using a latent diffusion process conditioned on both the low-resolution input
(concatenated in latent space) and a text prompt for guided detail generation.

Architecture components:

- U-Net with 7 input channels (4 latent + 3 low-res RGB conditioning)
- 320 base channels with [1, 2, 4, 4] multipliers (SD 2.x backbone)
- OpenCLIP ViT-H/14 text encoder (1024-dim cross-attention)
- Standard VAE (4-channel latent space, 0.18215 scale factor)
- DDIM scheduler for faster inference

Technical specifications:

- Architecture: U-Net with low-res RGB conditioning
- Input channels: 7 (4 latent noise + 3 low-res RGB)
- Output channels: 4 (latent space)
- Base channels: 320, multipliers [1, 2, 4, 4]
- Cross-attention dimension: 1024 (OpenCLIP ViT-H/14)
- 2 residual blocks per level
- Attention at 4x, 2x, and 1x downsampling
- Noise schedule: Linear beta [0.0001, 0.02], 1000 timesteps
- Default inference steps: 75
- Default guidance scale: 7.5
- Upscale factor: 4x

Reference: Rombach et al., "Stable Diffusion x4 Upscaler", Stability AI, 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDUpscalerModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of SDUpscalerModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `UpscaleFactor` | Gets the upscale factor (4x). |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image from a text prompt using SD Upscaler defaults. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs text-guided 4x image upscaling. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom or default configurations. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the U-Net backbone (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching OpenCLIP ViT-H/14 output (1024). |
| `DEFAULT_GUIDANCE_SCALE` | Default classifier-free guidance scale (7.5). |
| `DefaultHeight` | Default output image height (512). |
| `DefaultWidth` | Default output image width (512). |
| `INPUT_CHANNELS` | Total input channels for the U-Net (7 = 4 latent + 3 low-res RGB). |
| `LATENT_CHANNELS` | Number of latent channels (4). |
| `UPSCALE_FACTOR` | Upscale factor (4x). |
| `_conditioner` | The OpenCLIP text encoder conditioning module. |
| `_unet` | The U-Net noise predictor with low-resolution RGB conditioning. |
| `_vae` | The standard VAE for encoding/decoding between pixel and latent space. |

