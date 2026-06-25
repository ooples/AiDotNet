---
title: "MakeAVideoModel<T>"
description: "Make-A-Video model — text-to-video generation without paired text-video data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Make-A-Video model — text-to-video generation without paired text-video data.

## For Beginners

Make-A-Video is Meta's video generation model that creates videos
from text descriptions without needing paired text-video training data.

How Make-A-Video works:

1. Text prompt is encoded using CLIP and BPE into a 768-dimensional embedding
2. A pseudo-3D U-Net generates an initial low-resolution video in latent space
3. Temporal layers extend single images into coherent video sequences
4. Spatial and temporal super-resolution stages increase quality and frame rate

Advantages:

- Does not require paired text-video training data
- Leverages existing high-quality text-to-image knowledge
- Supports both text-to-video and image-to-video generation
- Pseudo-3D convolutions are more memory-efficient than full 3D

Limitations:

- Lower temporal consistency than full 3D attention models
- Limited to shorter video clips (16 frames default)
- Quality depends heavily on the underlying text-to-image model

## How It Works

Make-A-Video leverages text-to-image models and unsupervised video learning to
generate videos without requiring paired text-video training data. It uses a three-stage
pipeline: text-to-image base generation, temporal extension for motion, and spatial plus
temporal super-resolution for high-quality output.

Architecture components:

- Pseudo-3D U-Net with temporal convolutions and attention
- Standard VAE for spatial latent encoding (4-channel latent space)
- CLIP + BPE dual text encoder for conditioning (768-dim context)
- Three-stage cascade: base T2I, temporal extension, spatial/temporal SR

Technical specifications:

- Architecture: Pseudo-3D U-Net with temporal conv and attention
- Latent space: 4 channels with standard VAE
- Text encoder: CLIP + BPE (768-dimensional context)
- Base resolution: 64x64 latent (256x256 pixel equivalent)
- Super-resolution: up to 768x768 pixels
- Default: 16 frames at 8 FPS
- Noise schedule: DDPM linear

Reference: Singer et al., "Make-A-Video: Text-to-Video Generation without Text-Video Data", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MakeAVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of MakeAVideoModel with full customization support. |

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
| `InitializeLayers(VideoUNetPredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the video U-Net and VAE layers using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the U-Net backbone (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension from the CLIP text encoder (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Make-A-Video (7.5). |
| `LATENT_CHANNELS` | Number of latent channels for the standard VAE (4). |
| `NUM_HEADS` | Number of attention heads in the U-Net (8). |

