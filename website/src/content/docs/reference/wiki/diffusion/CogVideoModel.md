---
title: "CogVideoModel<T>"
description: "CogVideo/CogVideoX model for text-to-video and image-to-video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

CogVideo/CogVideoX model for text-to-video and image-to-video generation.

## For Beginners

CogVideoX generates videos from text descriptions:

How CogVideoX works:

1. Text is encoded by a T5 text encoder
2. A 3D causal VAE encodes/decodes video in compressed latent space
3. A diffusion transformer (DiT) denoises the video latents
4. The 3D VAE decodes latents back to video frames

Key characteristics:

- 3D causal VAE for temporal compression (4x temporal, 8x spatial)
- Expert transformer blocks with adaptive layer norm
- T5 text encoder for text understanding
- CogVideoX-2B: 2B parameters, 480p output
- CogVideoX-5B: 5B parameters, 720p output
- 49 frames at 8 FPS (~6 seconds)

Advantages:

- Open-source with permissive license
- Strong temporal coherence
- Good prompt adherence
- Efficient 3D VAE compression

Limitations:

- Generation is slow (~minutes per video)
- Requires significant VRAM
- Limited to short clips

## How It Works

CogVideoX is a large-scale text-to-video generation model developed by Zhipu AI / THUDM.
It uses a 3D causal VAE and a transformer-based architecture for generating coherent video.

Technical specifications:

- Architecture: 3D Causal VAE + Diffusion Transformer
- CogVideoX-2B: 2B parameters, 480×720
- CogVideoX-5B: 5B parameters, 480×720
- 3D Causal VAE: 4x temporal + 8x spatial compression, 16 latent channels
- Text encoder: T5-XXL (4096-dim)
- Frames: 49 at 8 FPS
- Noise schedule: Scaled linear, 1000 training timesteps

Reference: Hong et al., "CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CogVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,String,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of CogVideoModel with full customization support. |

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
| `Variant` | Gets the model variant ("2B" or "5B"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

