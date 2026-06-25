---
title: "ModelScopeT2VModel<T>"
description: "ModelScope Text-to-Video model with temporal U-Net for short video clip generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

ModelScope Text-to-Video model with temporal U-Net for short video clip generation.

## For Beginners

ModelScope T2V generates short video clips from text prompts.

How ModelScope T2V works:

1. Text prompt is encoded by CLIP into 1024-dim embeddings
2. Each video frame is encoded by the SD VAE into 4 latent channels
3. The temporal U-Net processes latent frames with temporal attention and convolution
4. Temporal attention ensures frames are consistent with each other
5. The VAE decodes each latent frame back to pixel space

Key characteristics:

- Based on SD 1.5 architecture with temporal blocks added
- Trained on WebVid-10M dataset (10M video-text pairs)
- 256x256 base resolution with cascaded upscaling to 512x512
- 16 frames at 8 FPS by default (~2 seconds)
- One of the first open-source text-to-video models

When to use ModelScope T2V:

- Simple text-to-video generation
- Research on temporal attention mechanisms
- Lightweight video generation on modest hardware
- Building on established SD 1.5 ecosystem

Limitations:

- Low resolution (256-512px)
- Short duration (16 frames)
- Quality below modern video models
- Single temporal attention layer limits motion coherence

## How It Works

ModelScope T2V extends the Stable Diffusion U-Net architecture with temporal convolution
and temporal attention modules, enabling text-to-video generation. It was one of the first
open-source text-to-video models trained on the WebVid-10M dataset.

Architecture components:

- Video U-Net based on SD 1.5 with temporal extension blocks
- 320 base channels with [1, 2, 4, 4] channel multipliers
- 1 temporal attention layer per block for inter-frame consistency
- CLIP text encoder for 1024-dim cross-attention conditioning
- Standard SD VAE for per-frame spatial compression (4 latent channels)
- DDPM noise schedule with scaled linear beta

Technical specifications:

- Architecture: Video U-Net (SD 1.5 + temporal blocks)
- Base channels: 320, multipliers [1, 2, 4, 4]
- Attention heads: 8
- ResNet blocks per level: 2
- Temporal attention layers: 1 per block
- Latent channels: 4 (standard SD VAE)
- Cross-attention dimension: 1024 (CLIP)
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Default: 16 frames at 8 FPS (~2 seconds)
- Training dataset: WebVid-10M

Reference: Wang et al., "ModelScope Text-to-Video Technical Report", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelScopeT2VModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of ModelScopeT2VModel with full customization support. |

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
| `InitializeLayers(VideoUNetPredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the Video U-Net and VAE layers using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the Video U-Net (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension from the CLIP text encoder (1024). |
| `DEFAULT_FPS` | Default frames per second (8). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (16). |
| `LATENT_CHANNELS` | Number of latent channels from the standard SD VAE (4). |
| `NUM_HEADS` | Number of attention heads (8). |
| `NUM_TEMPORAL_LAYERS` | Number of temporal attention layers per block (1). |
| `_conditioner` | The CLIP text encoder conditioning module. |
| `_vae` | The standard SD VAE for per-frame spatial compression. |
| `_videoUNet` | The Video U-Net noise predictor with temporal attention. |

