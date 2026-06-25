---
title: "KandinskyModel<T>"
description: "Kandinsky 2.2/3.0 model for text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Kandinsky 2.2/3.0 model for text-to-image generation.

## For Beginners

Kandinsky works in two phases, like a translator:

How Kandinsky works:

1. Prior stage: Translates text embeddings (CLIP) into image embeddings
2. Decoder stage: A latent diffusion model generates images conditioned on the image embeddings

Key characteristics:

- Two-stage pipeline: Prior → Decoder
- Prior: Diffusion-based mapping from text to image embedding space
- Decoder: U-Net latent diffusion model (similar to SD architecture)
- Text encoder: CLIP ViT-G/14 (1280-dim) + multilingual XLM-RoBERTa
- Image encoder: CLIP ViT-G/14 (used to train the prior)
- Native resolution: 1024x1024 (Kandinsky 3.0) or 512x512 (Kandinsky 2.2)
- VAE: Movq (MoVQ-GAN), 4 latent channels

Advantages:

- Strong multilingual support through XLM-RoBERTa
- Two-stage architecture allows separate optimization of text→embedding and embedding→image
- Good prompt adherence through CLIP-space prior

Limitations:

- Two-stage pipeline adds latency
- Smaller ecosystem than Stable Diffusion

## How It Works

Kandinsky is a two-stage text-to-image model developed by Sber AI and AI Forever.
It uses a prior model (diffusion or transformer) to map text embeddings to image embeddings,
then a latent diffusion decoder to generate images from those embeddings.

Technical specifications:

- Architecture: Prior + Latent Diffusion Decoder
- Prior: Diffusion transformer, maps CLIP text→image embeddings
- Decoder U-Net: ~1.2B parameters, 4-channel latent, channel multipliers [1, 2, 4, 4]
- Text encoder: CLIP ViT-G/14 (1280-dim)
- VAE: MoVQ-GAN, 4 latent channels, scale factor 0.18215
- Noise schedule: Linear beta schedule, 1000 training timesteps

Reference: Razzhigaev et al., "Kandinsky: an Improved Text-to-Image Synthesis with Image Prior and Latent Diffusion", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KandinskyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,KandinskyVersion,Nullable<Int32>)` | Initializes a new instance of KandinskyModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (1280 for CLIP ViT-G/14). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `PriorModel` | Gets the prior model that maps text embeddings to image embeddings. |
| `VAE` |  |
| `Version` | Gets the model version. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `InitializeLayers(UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the prior, decoder U-Net, and MoVQ-GAN VAE layers, using custom layers from the user if provided or creating industry-standard layers from the Kandinsky paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Kandinsky 3.0 generation. |
| `DefaultWidth` | Default image width for Kandinsky 3.0 generation. |
| `KANDINSKY_CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-G/14 output (1280). |
| `KANDINSKY_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Kandinsky (4.0). |
| `KANDINSKY_IMAGE_EMBEDDING_DIM` | Dimension of the CLIP image embedding space for the prior. |

