---
title: "HunyuanVideoModel<T>"
description: "HunyuanVideo model for dual-stream DiT video generation with unified image-video capability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

HunyuanVideo model for dual-stream DiT video generation with unified image-video capability.

## For Beginners

HunyuanVideo is Tencent's open-source video generation model.

How HunyuanVideo works:

1. Text prompt is encoded by an MLLM-based encoder into 4096-dim embeddings
2. Video is compressed by the 3D causal VAE into a 16-channel latent (4x4x4 compression)
3. The DS-DiT processes text and video in dual streams, then merges for joint attention
4. Flow matching denoises the video latent over the scheduled timesteps
5. The causal VAE decodes the latent back to 720p video

Key characteristics:

- 13B parameters total (one of the largest open-source video models)
- 720p resolution, 5+ second duration, 129 frames at 24 FPS
- Open-source weights available
- Unified image-video generation (both supported)
- 3D causal VAE enables temporal consistency

When to use HunyuanVideo:

- High-quality open-source video generation
- Text-to-video with strong prompt adherence
- Image-to-video animation
- Research and experimentation with large video models

Limitations:

- Very large model (13B parameters, requires significant GPU memory)
- Slower generation than lightweight models
- Causal VAE may have slight quality loss at clip boundaries

## How It Works

HunyuanVideo by Tencent uses a "Dual-stream to Single-stream" Diffusion Transformer
(DS-DiT) architecture with a 3D causal VAE for high-resolution video generation.
The dual-stream design processes text and video tokens separately in early layers,
then merges them for joint attention in later layers.

Architecture components:

- DS-DiT (Dual-stream to Single-stream DiT) with 40 transformer layers
- 3072 hidden dimension with 24 attention heads
- 3D causal VAE with 4x4x4 spatiotemporal compression (16 latent channels)
- MLLM-based text encoder for rich semantic understanding (4096-dim context)
- Flow matching training objective for stable convergence

Technical specifications:

- Architecture: DS-DiT (Dual-stream to Single-stream Diffusion Transformer)
- Hidden dimension: 3072
- Transformer layers: 40 (dual-stream early, single-stream late)
- Attention heads: 24
- Patch size: 2x2 spatiotemporal patches
- Latent channels: 16 (compressed by 3D causal VAE)
- Context dimension: 4096 (MLLM text encoder)
- VAE compression: 4x4x4 spatiotemporal
- Default: 129 frames at 24 FPS (~5.4 seconds)
- Training objective: Flow matching
- Total parameters: ~13B

Reference: Kong et al., "HunyuanVideo: A Systematic Framework For Large Video Generative Models", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HunyuanVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of HunyuanVideoModel with full customization support. |

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
| `CONTEXT_DIM` | Context dimension from the MLLM text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (129, ~5.4 seconds at 24 FPS). |
| `HIDDEN_DIM` | Hidden dimension of the DS-DiT transformer (3072). |
| `LATENT_CHANNELS` | Number of latent channels from the 3D causal VAE (16). |
| `NUM_HEADS` | Number of attention heads (24). |
| `NUM_LAYERS` | Number of transformer layers in the DS-DiT (40). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The MLLM-based text encoder conditioning module. |
| `_dit` | The DS-DiT noise predictor with dual-stream to single-stream architecture. |
| `_temporalVAE` | The 3D causal VAE for spatiotemporal video compression. |

