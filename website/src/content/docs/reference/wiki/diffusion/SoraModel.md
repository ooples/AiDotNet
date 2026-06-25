---
title: "SoraModel<T>"
description: "Sora-architecture model for DiT-based video generation with native spatiotemporal patches."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Sora-architecture model for DiT-based video generation with native spatiotemporal patches.

## For Beginners

Sora creates videos from text using a world-simulator approach.

How Sora works:

1. Text is encoded by dual CLIP + T5 encoders into 4096-dim embeddings
2. Video is compressed by the 3D causal VAE into 16-channel spatiotemporal patches
3. The DiT processes patches as a sequence (like tokens in a language model)
4. Full 3D attention captures spatial and temporal relationships simultaneously
5. Flow matching denoises the video over scheduled timesteps
6. The causal VAE decodes patches back to variable-duration video

Key characteristics:

- Native variable duration and resolution (no fixed grid)
- Trained on video + image data at native aspect ratios
- Full 3D attention (no factorization) for maximum quality
- "World simulator" approach for physically plausible generation
- 150 frames at 24 FPS by default (~6.25 seconds), up to 60s

When to use Sora:

- Highest-quality video generation
- Variable-length and multi-resolution content
- Physical simulation and world modeling
- Long-duration video generation

Limitations:

- Proprietary model (API-only access)
- Very large model with high compute requirements
- May struggle with complex physical interactions
- Generation can be slow for long videos

## How It Works

Sora by OpenAI uses a Diffusion Transformer (DiT) operating on spatiotemporal patches
of video, enabling native variable-duration, resolution, and aspect-ratio video generation.
The model treats videos as sequences of spacetime patches, similar to how LLMs process tokens.

Architecture components:

- DiT backbone with 48 transformer layers and 3072 hidden dimension
- 24 attention heads with full 3D spatiotemporal attention
- 3D spatiotemporal patch embeddings (patch size 2)
- 3D causal VAE with 16 latent channels for spatiotemporal compression
- 4096-dim text conditioning (CLIP + T5 dual encoder)
- Flow matching training objective

Technical specifications:

- Architecture: DiT with 3D spatiotemporal patches
- Hidden dimension: 3072
- Transformer layers: 48
- Attention heads: 24
- Patch size: 2 (spatiotemporal)
- Latent channels: 16 (3D causal VAE)
- Context dimension: 4096 (CLIP + T5 dual encoder)
- VAE: 3D causal with 3 temporal layers
- Default: 150 frames at 24 FPS (~6.25 seconds)
- Training objective: Flow matching
- Native variable resolution and duration

Reference: OpenAI, "Video generation models as world simulators", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoraModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of SoraModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
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
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the DiT and temporal VAE layers using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONTEXT_DIM` | Context dimension from the dual text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (150, ~6.25 seconds at 24 FPS). |
| `HIDDEN_DIM` | Hidden dimension of the DiT transformer (3072). |
| `LATENT_CHANNELS` | Number of latent channels from the 3D causal VAE (16). |
| `NUM_HEADS` | Number of attention heads (24). |
| `NUM_LAYERS` | Number of transformer layers (48). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The dual CLIP + T5 text encoder conditioning module. |
| `_dit` | The DiT noise predictor with full 3D spatiotemporal attention. |
| `_temporalVAE` | The 3D causal VAE for spatiotemporal video compression. |

