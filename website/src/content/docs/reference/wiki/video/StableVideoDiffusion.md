---
title: "StableVideoDiffusion<T>"
description: "Stable Video Diffusion (SVD) for image-to-video and text-to-video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Generation`

Stable Video Diffusion (SVD) for image-to-video and text-to-video generation.

## For Beginners

Stable Video Diffusion generates videos from images or text prompts.
It works by:

- Starting with random noise
- Gradually removing noise (denoising) over many steps
- Guided by the input image or text embedding

Key capabilities:

- Image-to-Video: Animate a still image into a short video clip
- Text-to-Video: Generate video from text descriptions
- Video extension: Continue an existing video
- Motion control: Adjust camera motion and subject movement

The model generates temporally consistent frames by processing spatial and
temporal attention together, ensuring smooth motion without flickering.

## How It Works

**Technical Details:**

- Based on latent diffusion in compressed video space
- 3D UNet with spatial and temporal attention layers
- CLIP text encoder for text conditioning
- VAE encoder/decoder for latent space compression
- Supports classifier-free guidance for quality control

**Reference:** Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets"
Stability AI, 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableVideoDiffusion(NeuralNetworkArchitecture<>,SVDModelVariant,Int32,Int32,Double,Int32,Int32,Int32,Int32,StableVideoDiffusionOptions)` |  |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `NumFrames` | Gets the number of output frames. |
| `OutputHeight` | Gets the output video height. |
| `OutputWidth` | Gets the output video width. |
| `SupportsTraining` | Gets whether training is supported. |
| `Variant` | Gets the model variant. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTemporalVariation(Tensor<>,Double,Int32,Int32,Int32,Int32)` | Adds temporal variation to create different frames from a single latent. |
| `AddTensors(Tensor<>,Tensor<>)` | Element-wise tensor addition for residual connections. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ExtendVideo(List<Tensor<>>,Int32,Nullable<Int32>)` | Extends an existing video by generating continuation frames. |
| `ExtractTemporalSlice(Tensor<>,Int32,Int32,Int32,Int32,Int32)` | Extracts a temporal slice from 5D latent tensor. |
| `GenerateFromImage(Tensor<>,Int32,Int32,Nullable<Int32>)` | Generates a video from an input image (image-to-video). |
| `GenerateFromText(Tensor<>,Int32,Int32,Nullable<Int32>)` | Generates a video from a text prompt. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `InterpolateKeyframes(List<Tensor<>>,Int32,Nullable<Int32>)` | Performs temporal interpolation between keyframes. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `TextEncoderFFN(Tensor<>,Int32)` | Feed-forward network for text encoder with GELU activation. |
| `TextEncoderLayerNorm(Tensor<>)` | Layer normalization for text encoder. |
| `TextEncoderSelfAttention(Tensor<>,Int32)` | Multi-head self-attention for text encoder following CLIP architecture. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultResolution` | Initializes a new instance of the StableVideoDiffusion class. |

