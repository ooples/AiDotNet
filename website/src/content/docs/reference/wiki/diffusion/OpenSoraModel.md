---
title: "OpenSoraModel<T>"
description: "Open-Sora model for open-source Sora-like video generation with STDiT architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Open-Sora model for open-source Sora-like video generation with STDiT architecture.

## For Beginners

Open-Sora is an open-source alternative to OpenAI's Sora.

How Open-Sora works:

1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
2. Video is compressed by the 3D causal VAE into 4-channel latent space
3. The STDiT applies spatial attention first, then temporal attention per layer
4. Rectified flow denoises the video latent over scheduled timesteps
5. The causal VAE decodes the latent back to video frames

Key characteristics:

- Open-source community project replicating Sora capabilities
- STDiT factorizes spatial and temporal attention for efficiency
- Rectified flow training (straighter trajectories than DDPM)
- Multi-resolution training enables variable aspect ratio/duration
- 51 frames at 24 FPS by default (~2.1 seconds)

When to use Open-Sora:

- Open-source video generation research
- Variable-resolution video generation
- Text-to-video, image-to-video, and video-to-video tasks
- Community-driven experimentation and fine-tuning

Limitations:

- Quality below commercial models (Sora, Veo)
- Research-stage model with ongoing improvements
- Limited duration compared to larger models
- Requires significant compute for training

## How It Works

Open-Sora is an open-source reproduction of Sora-like capabilities using the
Spatial-Temporal DiT (STDiT) architecture. It features efficient spatial-temporal
attention factorization, rectified flow training, and multi-resolution masking strategy.

Architecture components:

- STDiT (Spatial-Temporal Diffusion Transformer) with 28 layers
- 1152 hidden dimension with 16 attention heads
- 3D causal VAE with 4 latent channels for spatiotemporal compression
- T5-XXL text encoder for 4096-dim context embeddings
- Rectified flow training objective for efficiency
- Multi-resolution training with masking strategy

Technical specifications:

- Architecture: STDiT (Spatial-Temporal Diffusion Transformer)
- Hidden dimension: 1152
- Transformer layers: 28
- Attention heads: 16
- Patch size: 2
- Latent channels: 4 (3D causal VAE)
- Context dimension: 4096 (T5-XXL)
- Training objective: Rectified flow
- Multi-resolution training with masking
- Default: 51 frames at 24 FPS (~2.1 seconds)
- Supports: text-to-video, image-to-video, video-to-video

Reference: Zheng et al., "Open-Sora: Democratizing Efficient Video Production for All", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenSoraModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of OpenSoraModel with full customization support. |

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
| `InitializeLayers(DiTNoisePredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the STDiT and temporal VAE layers using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONTEXT_DIM` | Context dimension from the T5-XXL text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (51, ~2.1 seconds at 24 FPS). |
| `HIDDEN_DIM` | Hidden dimension of the STDiT transformer (1152). |
| `LATENT_CHANNELS` | Number of latent channels from the 3D causal VAE (4). |
| `NUM_HEADS` | Number of attention heads (16). |
| `NUM_LAYERS` | Number of transformer layers (28). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The T5-XXL text encoder conditioning module. |
| `_dit` | The STDiT noise predictor with spatial-temporal attention factorization. |
| `_temporalVAE` | The 3D causal VAE for spatiotemporal video compression. |

