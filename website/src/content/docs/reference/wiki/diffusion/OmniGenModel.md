---
title: "OmniGenModel<T>"
description: "OmniGen model — unified image generation model handling multiple tasks in one architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

OmniGen model — unified image generation model handling multiple tasks in one architecture.

## For Beginners

OmniGen is one model that does everything:

Key characteristics:

- Unified model: text-to-image, editing, inpainting, subject-driven, etc.
- No task-specific adapters needed (unlike ControlNet, IP-Adapter)
- Single transformer backbone handles all tasks
- Interleaved image-text input for flexible conditioning
- In-context learning: understands task from examples

Tasks OmniGen can handle:

- Text-to-image generation
- Image editing (instruction-based)
- Subject-driven generation (given reference images)
- Visual conditional generation (depth, edge, pose)
- Style transfer

Use OmniGen when you need:

- Single model for multiple generation tasks
- Simplified pipeline (no adapter management)
- Flexible conditioning from images and text

## How It Works

OmniGen is a unified image generation model that handles text-to-image, image editing,
subject-driven generation, and visual conditional generation with a single model,
without requiring task-specific adapters or fine-tuning.

Technical specifications:

- Architecture: Unified transformer with interleaved image-text tokens
- Parameters: ~3.8B
- Text encoder: integrated (not separate)
- Resolution: 512x512 to 1024x1024
- Latent channels: 4, 8x downsampling

Reference: Xiao et al., "OmniGen: Unified Image Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OmniGenModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of OmniGenModel with full customization support. |

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
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the DiT and VAE layers, using custom components if provided or creating industry-standard layers from the OmniGen specifications. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching the unified transformer's context size (2048). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for OmniGen (3.0, lower than typical due to unified architecture). |
| `DefaultHeight` | Default image height for OmniGen generation (1024 pixels). |
| `DefaultWidth` | Default image width for OmniGen generation (1024 pixels). |
| `LATENT_CHANNELS` | Number of latent channels in OmniGen's VAE (4 channels). |
| `_conditioner` | Optional conditioning module for text/image conditioning. |
| `_dit` | The DiT noise predictor (unified transformer backbone, ~3.8B parameters). |
| `_vae` | The VAE for encoding images to latent space and decoding back. |

