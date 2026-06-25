---
title: "RealESRGANModel<T>"
description: "Real-ESRGAN model for practical blind image super-resolution with degradation-aware training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

Real-ESRGAN model for practical blind image super-resolution with degradation-aware training.

## For Beginners

Real-ESRGAN upscales low-resolution real-world photos by 4x.

How Real-ESRGAN works:

1. The input low-resolution image is concatenated with the latent noise (8 input channels)
2. The U-Net predicts noise conditioned on the low-res image
3. The denoised latent is decoded to a high-resolution 4x upscaled image

Key characteristics:

- 4x upscaling factor (128x128 to 512x512, 256x256 to 1024x1024)
- Second-order degradation model handles realistic corruption chains
- Works well on faces, landscapes, anime, and general photography
- Unconditional by default (guidance scale 1.0, no text prompt needed)
- Can use text prompts for guided upscaling when conditioner is provided

When to use Real-ESRGAN:

- Upscaling low-resolution photos from the web or social media
- Enhancing old photographs and scanned images
- Anime and illustration upscaling (specialized models available)
- Batch processing of image libraries

Limitations:

- Fixed 4x upscale factor (use SDUpscaler for flexible scaling)
- May add unnatural sharpness to already-sharp images
- Large model size due to RRDB backbone

## How It Works

Real-ESRGAN combines the ESRGAN architecture with a second-order degradation model
for practical blind super-resolution that handles complex real-world degradations
including blur, noise, JPEG artifacts, and their combinations.

Architecture components:

- RRDB-Net backbone (Residual-in-Residual Dense Blocks) with 23 blocks
- U-Net discriminator with spectral normalization for training stability
- Second-order degradation model simulating real-world image corruption
- Diffusion-based refinement using concatenated low-res conditioning (8 input channels)
- Standard VAE (4-channel latent space, 0.18215 scale factor)

Technical specifications:

- Architecture: RRDB-Net + diffusion refinement with low-res conditioning
- Input channels: 8 (4 latent + 4 downscaled low-res conditioning)
- Output channels: 4 (latent space)
- Base channels: 128
- Channel multipliers: [1, 2, 4]
- Upscale factor: 4x
- RRDB blocks: 23 (in the full ESRGAN backbone)
- Noise schedule: Linear beta [0.0001, 0.02], 1000 timesteps
- Default guidance scale: 1.0 (unconditional)

Reference: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCVW 2021

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealESRGANModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of RealESRGANModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` | Counts the flat-API parameter surface (predictor + VAE). |
| `UpscaleFactor` | Gets the upscale factor (4x). |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image from a text prompt using Real-ESRGAN defaults. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs 4x image upscaling using the Real-ESRGAN diffusion pipeline. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom or default configurations. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the Real-ESRGAN U-Net (128). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension for optional text conditioning (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Real-ESRGAN (1.0, unconditional). |
| `DefaultHeight` | Default image height for Real-ESRGAN output. |
| `DefaultWidth` | Default image width for Real-ESRGAN output. |
| `INPUT_CHANNELS` | Input channels for the U-Net (8 = 4 latent + 4 downscaled low-res). |
| `LATENT_CHANNELS` | Number of latent channels (4). |
| `UPSCALE_FACTOR` | Upscale factor for Real-ESRGAN (4x). |
| `_conditioner` | Optional text conditioning module for guided upscaling. |
| `_unet` | The U-Net noise predictor with low-resolution image conditioning. |
| `_vae` | The standard VAE for encoding/decoding between pixel and latent space. |

