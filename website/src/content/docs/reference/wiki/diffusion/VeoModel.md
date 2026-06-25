---
title: "VeoModel<T>"
description: "Veo model for Google's high-fidelity cascaded video generation with temporal super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Veo model for Google's high-fidelity cascaded video generation with temporal super-resolution.

## For Beginners

Veo is Google's top-tier video generation model.

How Veo works:

1. Text is encoded by dual T5-XXL + CLIP encoders into 4096-dim embeddings
2. Base model generates low-resolution video in compressed latent space
3. Spatial super-resolution stage upscales each frame
4. Temporal super-resolution stage adds interpolated frames
5. The causal VAE decodes the final high-resolution video

Key characteristics:

- Cascaded architecture: base → spatial SR → temporal SR
- 1080p output with 60+ second duration capability
- Dual text encoding (T5-XXL + CLIP) for rich conditioning
- Veo 2 variant with improved quality and consistency
- 150 frames at 24 FPS by default (~6.25 seconds)

When to use Veo:

- Highest-quality video generation
- Long-duration video content
- 1080p high-resolution output
- Text-to-video, image-to-video, and video-to-video tasks

Limitations:

- Proprietary model (API-only access through Google)
- Very high compute requirements for cascaded generation
- Slower generation due to multi-stage pipeline
- Limited public information on exact architecture details

## How It Works

Veo by Google DeepMind uses cascaded diffusion with temporal super-resolution for
high-resolution, long-duration video generation. The base model generates at lower
resolution, then spatial and temporal super-resolution stages produce the final output.

Architecture components:

- Cascaded DiT with 40 transformer layers and 2560 hidden dimension
- 20 attention heads with full spatiotemporal attention
- T5-XXL + CLIP dual text encoding for 4096-dim context
- 3D causal VAE with 16 latent channels and 3 temporal layers
- Cascaded pipeline: base → spatial SR → temporal SR
- Flow matching training objective

Technical specifications:

- Architecture: Cascaded DiT (base + spatial SR + temporal SR)
- Hidden dimension: 2560
- Transformer layers: 40
- Attention heads: 20
- Patch size: 2
- Latent channels: 16 (3D causal VAE)
- Context dimension: 4096 (T5-XXL + CLIP dual encoder)
- Output resolution: Up to 1080p
- Default: 150 frames at 24 FPS (~6.25 seconds)
- Veo 2: Enhanced quality variant (200 frames default)
- Training objective: Flow matching

Reference: Google DeepMind, "Veo: High-Fidelity Video Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VeoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Boolean,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of VeoModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsVeo2` | Gets whether this is a Veo 2 variant with enhanced quality. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsImageToVideo` |  |
| `SupportsTextToVideo` |  |
| `SupportsVideoToVideo` |  |
| `TemporalVAE` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateVeo2(IConditioningModule<>)` | Creates a Veo 2 variant with enhanced quality and longer duration. |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the cascaded DiT and temporal VAE using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONTEXT_DIM` | Context dimension from the dual text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames for Veo (150). |
| `HIDDEN_DIM` | Hidden dimension of the cascaded DiT (2560). |
| `LATENT_CHANNELS` | Number of latent channels from the 3D causal VAE (16). |
| `NUM_HEADS` | Number of attention heads (20). |
| `NUM_LAYERS` | Number of transformer layers (40). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The dual T5-XXL + CLIP text encoder conditioning module. |
| `_dit` | The cascaded DiT noise predictor. |
| `_isVeo2` | Whether this is a Veo 2 variant. |
| `_temporalVAE` | The 3D causal VAE for spatiotemporal video compression. |

