---
title: "KlingModel<T>"
description: "Kling model — 3D spatiotemporal attention video generation by Kuaishou."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Kling model — 3D spatiotemporal attention video generation by Kuaishou.

## For Beginners

Kling is a video generation model from Kuaishou that creates
high-quality videos from text or images.

How Kling works:

1. Text prompt is encoded into a 4096-dimensional embedding
2. A 3D DiT transformer generates video in compressed latent space
3. Full 3D attention (not factorized) captures spatial AND temporal relationships
4. A temporal VAE decompresses the latent video into pixel-space frames

Advantages:

- Very high video quality at 1080p resolution
- Long video support (up to 2 minutes)
- Strong physics and motion understanding
- Both text-to-video and image-to-video generation

Limitations:

- Very large model requiring significant GPU memory
- Proprietary architecture with limited public details
- Slower generation than smaller video models

## How It Works

Kling uses a 3D VAE with full spatiotemporal attention for high-quality video generation
with strong motion consistency and physics understanding. It supports up to 2 minutes of
video at 1080p resolution.

Architecture components:

- DiT backbone with 3D full attention (36 layers, 2048 hidden dim)
- Temporal VAE for spatiotemporal compression (causal mode)
- Large-scale text encoder (4096-dim context)
- 16-channel latent space for high-fidelity reconstruction

Technical specifications:

- Architecture: DiT with full 3D spatiotemporal attention
- Parameters: ~5B+ (estimated)
- Backbone: 36 transformer layers, 2048 hidden dim, 16 attention heads
- Latent space: 16 channels with temporal VAE
- Text encoder: 4096-dimensional context embedding
- Max resolution: 1080p (1920x1080)
- Max duration: 2 minutes at 30 FPS
- Noise schedule: Flow matching

Reference: Kuaishou, "Kling: A Text-to-Video Generation Model", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KlingModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of KlingModel with full customization support. |

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
| `CONTEXT_DIM` | Context dimension from the text encoder (4096). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Kling (7.0). |
| `HIDDEN_DIM` | Hidden dimension of the DiT backbone (2048). |
| `LATENT_CHANNELS` | Number of latent channels for Kling's temporal VAE (16 for high fidelity). |
| `NUM_HEADS` | Number of attention heads in the DiT backbone (16). |
| `NUM_LAYERS` | Number of transformer layers in the DiT backbone (36). |

