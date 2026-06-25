---
title: "LTXVideoModel<T>"
description: "LTX-Video model for lightweight real-time video generation with extreme latent compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

LTX-Video model for lightweight real-time video generation with extreme latent compression.

## For Beginners

LTX-Video generates videos faster than real-time by compressing heavily.

How LTX-Video works:

1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
2. Video is compressed by the 3D causal VAE into 128-channel latent (192x compression)
3. The lightweight DiT denoises the latent using flow matching
4. The causal VAE decodes the latent back to 720p video

Key characteristics:

- ~2B parameters (lightweight for a video model)
- 192x spatiotemporal compression (highest in class)
- 128 latent channels for rich representations despite compression
- 720p, 5 seconds at 24 FPS, faster-than-real-time generation
- Open-source with Lightricks weights

When to use LTX-Video:

- Real-time or interactive video generation
- Applications requiring low latency
- Edge deployment with limited compute
- Rapid prototyping and iteration

Limitations:

- High compression may lose fine spatial details
- Quality may not match larger models (Sora, HunyuanVideo)
- Best for medium-resolution content
- Trade-off between speed and fidelity

## How It Works

LTX-Video by Lightricks is designed for efficient video generation, using a lightweight
DiT transformer operating in a highly compressed latent space via a 3D causal VAE.
The extreme 192x compression ratio enables faster-than-real-time generation.

Architecture components:

- Lightweight DiT with 28 transformer layers and 1536 hidden dimension
- 16 attention heads for multi-head self-attention
- 3D causal VAE with 128 latent channels and 192x compression ratio
- T5-XXL text encoder for 4096-dim context embeddings
- Flow matching training objective for stable convergence
- Patch size 1 (operates on individual latent tokens)

Technical specifications:

- Architecture: Lightweight DiT with 3D causal VAE
- Hidden dimension: 1536
- Transformer layers: 28
- Attention heads: 16
- Patch size: 1 (direct latent token processing)
- Latent channels: 128 (extreme compression)
- Context dimension: 4096 (T5-XXL)
- VAE compression: 192x spatiotemporal
- Default: 121 frames at 24 FPS (~5 seconds)
- Training objective: Flow matching
- Total parameters: ~2B

Reference: Lightricks, "LTX-Video: Realtime Video Latent Diffusion", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LTXVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of LTXVideoModel with full customization support. |

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
| `CONTEXT_DIM` | Context dimension from the T5-XXL text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (121, ~5 seconds at 24 FPS). |
| `HIDDEN_DIM` | Hidden dimension of the lightweight DiT (1536). |
| `LATENT_CHANNELS` | Number of latent channels from the 3D causal VAE (128, extreme compression). |
| `NUM_HEADS` | Number of attention heads (16). |
| `NUM_LAYERS` | Number of transformer layers (28). |
| `PATCH_SIZE` | Patch size (1, operates on individual latent tokens). |
| `_conditioner` | The T5-XXL text encoder conditioning module. |
| `_dit` | The lightweight DiT noise predictor. |
| `_temporalVAE` | The 3D causal VAE with 192x compression. |

