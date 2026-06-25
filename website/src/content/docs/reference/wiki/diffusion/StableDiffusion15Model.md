---
title: "StableDiffusion15Model<T>"
description: "Stable Diffusion 1.5 model for text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Stable Diffusion 1.5 model for text-to-image generation.

## For Beginners

Stable Diffusion 1.5 is the "classic" text-to-image AI model.

How SD 1.5 works:

1. Your text prompt is encoded by a CLIP ViT-L/14 text encoder into embeddings
2. These embeddings guide a U-Net (865M parameters) that denoises in latent space
3. A VAE decodes the denoised latent into a 512x512 image

Key characteristics:

- Single text encoder: CLIP ViT-L/14 (768-dim embeddings)
- U-Net: 865M parameters, channel multipliers [1, 2, 4, 4]
- VAE: 4-channel latent space, 8x spatial downsampling
- Native resolution: 512x512 pixels
- Latent scale factor: 0.18215
- Guidance scale: 7.5 (default)

When to use SD 1.5:

- Huge community model ecosystem (thousands of fine-tunes available)
- Lower resource requirements than SDXL (runs on 4GB+ VRAM)
- Fast generation (20-50 steps)
- Excellent for 512x512 generation

Limitations:

- Lower resolution than SDXL (512x512 vs 1024x1024)
- Single text encoder (less prompt understanding than dual-encoder models)
- Occasional artifacts at hands, text, and complex scenes

## How It Works

Stable Diffusion 1.5 (SD 1.5) is a latent diffusion model developed by Stability AI and
Runway ML. It is the most widely used open-source text-to-image model and the foundation
for an enormous ecosystem of fine-tunes, LoRAs, ControlNets, and community models.

Technical specifications:

- Architecture: U-Net with cross-attention and time embedding
- Text encoder: CLIP ViT-L/14 (63M parameters, 768-dim, 77 max tokens)
- U-Net: 865M parameters, base channels 320, [1, 2, 4, 4] multipliers
- Cross-attention dimension: 768 (matches CLIP output)
- Attention resolutions: at 4x, 2x, and 1x downsampling levels
- VAE: KL-regularized autoencoder, 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta schedule, 1000 training timesteps
- Beta range: [0.00085, 0.012]
- Prediction type: Epsilon (noise prediction)

Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableDiffusion15Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of StableDiffusion15Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (768 for SD 1.5, matching CLIP ViT-L/14). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `EnsureParameterShapesResolved` | Materializes lazy submodule weights before state-dict style operations. |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image from a text prompt using SD 1.5 defaults. |
| `GenerateVariations(String,String,Int32,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates multiple image variations from the same prompt using different seeds. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs image-to-image transformation using SD 1.5. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers, using custom layers from the user if provided or creating industry-standard layers from the SD 1.5 research paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for SD 1.5 generation. |
| `DefaultWidth` | Default image width for SD 1.5 generation. |
| `SD15_CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-L/14 output (768). |
| `SD15_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for SD 1.5 (7.5). |
| `SD15_LATENT_CHANNELS` | Number of latent channels in SD 1.5's VAE. |
| `SD15_VAE_SCALE_FACTOR` | Spatial downsampling factor of the VAE (512 / 8 = 64). |
| `_conditioner` | The CLIP text encoder conditioning module. |
| `_parameterShapesResolved` | Tracks whether lazy UNet/VAE parameter shapes have been materialized. |
| `_unet` | The U-Net noise predictor (865M parameters in the full model). |
| `_vae` | The VAE for encoding images to latent space and decoding back. |

