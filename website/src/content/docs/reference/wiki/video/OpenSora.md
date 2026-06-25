---
title: "OpenSora<T>"
description: "OpenSora - Open-source Sora-like video generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Generation`

OpenSora - Open-source Sora-like video generation model.

## For Beginners

OpenSora generates videos from text descriptions, similar to how
image generation models like DALL-E or Stable Diffusion work but for videos.

Key capabilities:

- Text-to-Video: Generate videos from text descriptions
- Image-to-Video: Animate still images
- Video continuation: Extend existing videos
- Variable length: Generate videos of different durations
- Multiple aspect ratios: Support various video dimensions

Example prompts:

- "A cat playing with a ball in a sunny garden"
- "Time-lapse of a flower blooming"
- "A spaceship flying through an asteroid field"

## How It Works

**Technical Details:**

- Spatiotemporal DiT (Diffusion Transformer) architecture
- Variable resolution and duration support
- Efficient 3D attention mechanisms
- Progressive training strategy

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenSora` | Creates a default OpenSora video generation model with small default dimensions. |
| `OpenSora(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double,OpenSoraOptions)` | Creates an OpenSora video generation model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `NumFrames` | Gets the number of frames to generate. |
| `OutputHeight` | Gets the output frame height. |
| `OutputWidth` | Gets the output frame width. |
| `SupportsTraining` | Gets whether training is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeserializeNetworkSpecificData(BinaryReader)` | Restores model configuration from serialized data. |
| `DiTMultiHeadAttention(Tensor<>,Int32[])` | Applies multi-head self-attention for DiT blocks following the Transformer architecture. |
| `EncodeImage(Tensor<>)` | Encodes an image to latent space using a learned VAE encoder. |
| `ExtendVideo(List<Tensor<>>,Tensor<>,Nullable<Int32>)` | Extends an existing video. |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection for direct access. |
| `FallbackEncodeImage(Tensor<>)` | Fallback image encoding using simple average pooling when learned encoder is not available. |
| `GenerateCustom(Tensor<>,Int32,Int32,Int32,Nullable<Int32>)` | Generates video with custom duration and aspect ratio. |
| `GenerateFromImage(Tensor<>,Tensor<>,Nullable<Int32>)` | Generates a video from an image (image-to-video). |
| `GenerateFromText(Tensor<>,Nullable<Int32>)` | Generates a video from a text prompt. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for OpenSora. |
| `LayerNorm(Tensor<>)` | Applies layer normalization (standardization) across spatial dimensions. |
| `PredictCore(Tensor<>)` | Performs a single denoising prediction step on the input latents. |
| `Train(Tensor<>,Tensor<>)` | Trains the model using the diffusion training objective. |
| `UnpatchifyNoise(Tensor<>,Int32[])` | Converts patched noise back to full resolution using pixel shuffle and bilinear interpolation. |

