---
title: "DiffBIRModel<T>"
description: "DiffBIR model for blind image restoration with generative diffusion prior."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

DiffBIR model for blind image restoration with generative diffusion prior.

## For Beginners

DiffBIR restores degraded images (blur, noise, JPEG artifacts, old photos).

How DiffBIR works:

1. Stage 1 (SwinIR): A regression network removes obvious degradation from the input
2. Stage 2 (Diffusion): The SD prior adds realistic details to the cleaned image
3. LAControlNet: Controls how much the diffusion stage can modify the Stage 1 output

Key characteristics:

- Handles blind (unknown) degradation types automatically
- Two-stage design separates degradation removal from detail generation
- Controllable balance between fidelity to input and generated quality
- Works for face restoration, general images, and old photo restoration
- Based on Stable Diffusion 1.5 backbone for high-quality detail generation

When to use DiffBIR:

- Restoring old or damaged photographs
- Removing JPEG compression artifacts
- Denoising and deblurring real-world images
- Face restoration in group photos or surveillance footage

Limitations:

- Two-stage pipeline is slower than single-stage methods
- May hallucinate details not present in the original image
- Best results at 512x512 (SD 1.5 native resolution)

## How It Works

DiffBIR (Diffusion-Based Blind Image Restoration) uses a two-stage pipeline for
real-world image restoration. The first stage removes degradation using a regression
module (SwinIR), and the second stage refines details using a Stable Diffusion prior
with a controllable module for balancing fidelity and quality.

Architecture components:

- Stage 1: SwinIR regression module for degradation removal
- Stage 2: SD 1.5 U-Net backbone (320 base channels, [1,2,4,4] multipliers)
- Controllable feature wrapping module (LAControlNet) for fidelity-quality balance
- Standard VAE (4-channel latent space, 0.18215 scale factor)
- CLIP ViT-L/14 text encoder (768-dim cross-attention)

Technical specifications:

- Architecture: Two-stage (SwinIR + SD 1.5 U-Net with LAControlNet)
- Stage 1: SwinIR with 6 RSTB blocks, 180 channels, window size 8
- Stage 2: SD 1.5 U-Net backbone, 320 base channels, [1,2,4,4] multipliers
- Cross-attention dimension: 768 (CLIP ViT-L/14)
- VAE: 4 latent channels, scale factor 0.18215, 8x downsampling
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Default guidance scale: 7.5
- Default resolution: 512x512

Reference: Lin et al., "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior", ECCV 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffBIRModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of DiffBIRModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (768, matching CLIP ViT-L/14). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image from a text prompt using DiffBIR defaults. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs image restoration using the DiffBIR diffusion pipeline. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers, using custom components if provided or creating industry-standard layers from the DiffBIR research paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the SD 1.5 U-Net backbone (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-L/14 output (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default classifier-free guidance scale for DiffBIR (7.5). |
| `DefaultHeight` | Default image height for DiffBIR restoration. |
| `DefaultWidth` | Default image width for DiffBIR restoration. |
| `LATENT_CHANNELS` | Number of latent channels in the standard SD VAE (4). |
| `VAE_SCALE_FACTOR` | VAE spatial downsampling factor (8x). |
| `_conditioner` | The CLIP text encoder conditioning module for prompt-guided restoration. |
| `_unet` | The U-Net noise predictor using the SD 1.5 backbone architecture. |
| `_vae` | The standard VAE for encoding images to latent space and decoding back. |

