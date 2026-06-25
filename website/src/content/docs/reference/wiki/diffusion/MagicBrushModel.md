---
title: "MagicBrushModel<T>"
description: "MagicBrush model for instruction-based image editing with visual brush stroke guidance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

MagicBrush model for instruction-based image editing with visual brush stroke guidance.

## For Beginners

MagicBrush lets you edit images by describing what you want changed.

How it works:

1. You provide a source image and a text instruction (e.g., "make the sky sunset-colored")
2. Optionally provide brush strokes to guide where edits should occur
3. The model performs the described edit while preserving the rest of the image

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- Trained on large-scale human-annotated editing pairs
- Supports both instruction-only and brush-guided editing
- Uses Euler scheduler for fast, high-quality sampling

## How It Works

MagicBrush enables instruction-guided image editing by combining natural language instructions
with visual brush stroke annotations. It was trained on a large-scale dataset of manually
annotated editing triplets (source image, instruction, target image) collected using
DALL-E 2 and human annotators, allowing it to perform precise local and global edits.

Technical specifications:

- Architecture: SD 1.5 U-Net fine-tuned for instruction-based editing
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: Euler discrete

Reference: Zhang et al., "MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MagicBrushModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of MagicBrushModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (768 for CLIP ViT-L/14). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom layers if provided, or creating industry-standard SD 1.5 layers. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-L/14 output (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default classifier-free guidance scale (7.5). |
| `DefaultHeight` | Default image height for MagicBrush (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for MagicBrush (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

