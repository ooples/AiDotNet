---
title: "SUPIRModel<T>"
description: "SUPIR model for scaling up image restoration with SDXL for photo-realistic results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

SUPIR model for scaling up image restoration with SDXL for photo-realistic results.

## For Beginners

SUPIR uses SDXL (the most capable open-source image model)
combined with GPT language understanding for the highest quality image restoration.

How SUPIR works:

1. A GPT model analyzes the degraded image and generates a quality description
2. The description guides SDXL's diffusion process for intelligent restoration
3. SDXL's 2048-dim dual text encoder provides rich semantic conditioning
4. The result is photo-realistic detail at 1024x1024 native resolution

Key characteristics:

- Based on SDXL (1024x1024 native resolution, much higher than SD 1.5's 512x512)
- GPT-guided semantic understanding of image content and degradation
- Dual text encoders (2048-dim combined) for precise conditioning
- DPM-Solver++ for efficient 50-step inference
- SDXL VAE scale factor (0.13025, different from SD 1.5's 0.18215)

When to use SUPIR:

- Highest possible quality restoration is needed
- Processing at 1024x1024 resolution
- Complex degradation where semantic understanding helps
- Photo-realistic detail generation for important images

Limitations:

- Requires significantly more VRAM than SD 1.5-based models (~12GB+)
- Slower than lighter SR methods due to SDXL backbone
- GPT component adds complexity and potential dependency

## How It Works

SUPIR (Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image
Restoration In the Wild) leverages the SDXL model's generative prior with a GPT-guided
restoration pipeline for photo-realistic super-resolution at high resolutions. It combines
semantic understanding from a large language model with SDXL's generation quality.

Architecture components:

- SDXL U-Net backbone (320 base channels, [1, 2, 4] multipliers)
- Dual text encoders: OpenCLIP ViT-bigG/14 + CLIP ViT-L/14 (2048-dim combined)
- GPT-guided quality description for semantic restoration understanding
- SDXL VAE (4-channel latent, 0.13025 scale factor for SDXL)
- DPM-Solver++ multistep scheduler for efficient inference

Technical specifications:

- Architecture: SDXL U-Net + GPT-guided conditioning
- U-Net: 320 base channels, multipliers [1, 2, 4] (SDXL config)
- Cross-attention dimension: 2048 (dual OpenCLIP + CLIP)
- VAE: 4 latent channels, SDXL scale factor 0.13025
- Native resolution: 1024x1024 pixels
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Default scheduler: DPM-Solver++ Multistep
- Default guidance scale: 7.5

Reference: Yu et al., "Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SUPIRModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of SUPIRModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image from a text prompt using SUPIR/SDXL defaults. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs SDXL-quality image restoration using SUPIR. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom or default configurations. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the SDXL U-Net backbone (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension for SDXL dual text encoders (2048). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for SUPIR (7.5). |
| `DefaultHeight` | Default image height for SUPIR output (1024, SDXL native). |
| `DefaultWidth` | Default image width for SUPIR output (1024, SDXL native). |
| `LATENT_CHANNELS` | Number of latent channels (4). |
| `SDXL_VAE_SCALE_FACTOR` | SDXL VAE scale factor (0.13025). |
| `_conditioner` | Optional dual text encoder conditioning module. |
| `_unet` | The U-Net noise predictor using the SDXL backbone architecture. |
| `_vae` | The SDXL VAE for encoding/decoding between pixel and latent space. |

