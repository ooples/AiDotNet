---
title: "WanVideoModel<T>"
description: "Wan video model for Alibaba's scalable DiT video generation with full 3D attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Wan video model for Alibaba's scalable DiT video generation with full 3D attention.

## For Beginners

Wan generates high-quality videos with multiple size variants.

How Wan works:

1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
2. Video is compressed by WanVAE into 16-channel latent with causal temporal compression
3. The scalable DiT processes all patches with full 3D attention (no shortcuts)
4. Full 3D attention captures all spatial-temporal relationships simultaneously
5. Flow matching denoises the video over scheduled timesteps
6. WanVAE decodes the latent back to video frames

Key characteristics:

- Three scale variants: 1.3B (fast), 5B (balanced), 14B (highest quality)
- Full 3D attention (no factorization) for maximum quality at each scale
- WanVAE: specialized causal 3D VAE with 3 temporal layers
- Text-to-video and image-to-video support
- Open-source weights available
- 81 frames at 16 FPS by default (~5 seconds)

When to use Wan:

- Scalable video generation (choose variant for quality/speed trade-off)
- High-quality text-to-video generation
- Image animation (image-to-video)
- Research on scalable video architectures

Limitations:

- 14B variant requires significant GPU memory
- 1.3B variant sacrifices quality for speed
- Full 3D attention is O(n^2) in sequence length
- Limited video duration compared to cascaded models

## How It Works

Wan by Alibaba uses a scalable DiT architecture with full 3D attention (no factorization)
and a specialized WanVAE for temporally compressed video generation. The model supports
multiple scale variants: 1.3B, 5B, and 14B parameters.

Architecture components:

- Scalable DiT with full 3D attention (no spatial-temporal factorization)
- 1.3B variant: 1536 hidden, 30 layers, 12 heads
- 5B variant: 2560 hidden, 36 layers, 20 heads
- 14B variant: 3072 hidden, 40 layers, 24 heads (default)
- WanVAE: specialized causal 3D VAE with 16 latent channels
- T5-XXL text encoder for 4096-dim context embeddings
- Flow matching training objective

Technical specifications:

- Architecture: Scalable DiT with full 3D attention
- 1.3B: 1536 hidden, 30 layers, 12 heads (~1.3B parameters)
- 5B: 2560 hidden, 36 layers, 20 heads (~5B parameters)
- 14B: 3072 hidden, 40 layers, 24 heads (~14B parameters)
- Patch size: 2 (all variants)
- Latent channels: 16 (WanVAE)
- Context dimension: 4096 (T5-XXL)
- WanVAE: causal 3D VAE, 3 temporal layers, kernel size 3
- Default: 81 frames at 16 FPS (~5 seconds)
- Training objective: Flow matching
- Open-source: Yes

Reference: Alibaba, "Wan: Open and Advanced Large-Scale Video Generative Models", 2025

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WanVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,String,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of WanVideoModel with full customization support. |

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
| `Variant` | Gets the model variant identifier (1.3B, 5B, or 14B). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Create1_3B(IConditioningModule<>)` | Creates a 1.3B lightweight variant for fast generation. |
| `Create5B(IConditioningModule<>)` | Creates a 5B medium variant balancing quality and speed. |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `GetVariantConfig(String)` | Gets the architecture configuration for a given variant. |
| `InitializeLayers(DiTNoisePredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the DiT and WanVAE layers using custom or variant-appropriate defaults. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONTEXT_DIM` | Context dimension from the T5-XXL text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (16). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (81, ~5 seconds at 16 FPS). |
| `LATENT_CHANNELS` | Number of latent channels from the WanVAE (16). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The T5-XXL text encoder conditioning module. |
| `_dit` | The scalable DiT noise predictor with full 3D attention. |
| `_numHeads` | Number of attention heads for the current variant. |
| `_temporalVAE` | The WanVAE (causal 3D VAE) for temporally compressed video encoding/decoding. |
| `_variant` | The model variant identifier (1.3B, 5B, or 14B). |

