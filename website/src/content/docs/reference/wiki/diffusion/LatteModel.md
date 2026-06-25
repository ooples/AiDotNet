---
title: "LatteModel<T>"
description: "Latte model for Latent Diffusion Transformer video generation with factorized spatial-temporal attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Latte model for Latent Diffusion Transformer video generation with factorized spatial-temporal attention.

## For Beginners

Latte applies DiT (Diffusion Transformer) concepts to video generation.

How Latte works:

1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
2. Each video frame is encoded by the SD VAE into 4 latent channels
3. Latent frames are patchified and processed by the DiT with factorized attention
4. Spatial attention handles per-frame content, temporal attention handles motion
5. The VAE decodes each latent frame back to pixel space

Key characteristics:

- Explores 4 attention decomposition strategies for efficiency
- "Decomposed" variant (spatial then temporal) achieves best quality/speed
- Uses per-frame VAE (standard SD VAE, not temporal)
- 16 frames at 8 FPS by default (~2 seconds)
- ~700M parameter DiT backbone

When to use Latte:

- Research on efficient video DiT architectures
- Short clip generation from text prompts
- Exploring spatial-temporal attention decomposition
- Lightweight alternative to full 3D attention models

Limitations:

- Per-frame VAE may cause minor temporal artifacts
- Shorter duration than temporal-VAE-based models
- Lower resolution than SDXL-based video models
- Research model, not production-grade quality

## How It Works

Latte applies Diffusion Transformer (DiT) architecture to video generation by exploring
factorized spatial-temporal attention patterns within transformer blocks. The model
decomposes full 3D attention into efficient spatial and temporal components.

Architecture components:

- DiT backbone with 28 transformer layers and 1152 hidden dimension
- 4 attention variants: joint, spatial-first, temporal-first, decomposed
- 16 attention heads with efficient O(n) factorized attention
- T5-XXL text encoder for 4096-dim context embeddings
- Standard SD VAE for per-frame spatial compression (4 latent channels)
- Patch size 2 for spatiotemporal tokenization

Technical specifications:

- Architecture: DiT with factorized spatial-temporal attention
- Hidden dimension: 1152
- Transformer layers: 28
- Attention heads: 16
- Patch size: 2
- Latent channels: 4 (standard SD VAE)
- Context dimension: 4096 (T5-XXL)
- Default: 16 frames at 8 FPS (~2 seconds)
- Noise schedule: Linear beta [0.0001, 0.02], 1000 timesteps
- Scheduler: DDIM
- Parameters: ~700M (DiT backbone)

Reference: Ma et al., "Latte: Latent Diffusion Transformer for Video Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LatteModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of LatteModel with full customization support. |

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
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the DiT and VAE layers using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONTEXT_DIM` | Context dimension from the T5-XXL text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (8). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (16). |
| `HIDDEN_DIM` | Hidden dimension of the DiT transformer (1152). |
| `LATENT_CHANNELS` | Number of latent channels from the standard SD VAE (4). |
| `NUM_HEADS` | Number of attention heads (16). |
| `NUM_LAYERS` | Number of transformer layers (28). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The T5-XXL text encoder conditioning module. |
| `_dit` | The DiT noise predictor with factorized spatial-temporal attention. |
| `_vae` | The standard SD VAE for per-frame spatial compression. |

