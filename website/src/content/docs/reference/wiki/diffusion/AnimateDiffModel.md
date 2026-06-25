---
title: "AnimateDiffModel<T>"
description: "AnimateDiff model for text-to-video and image-to-video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

AnimateDiff model for text-to-video and image-to-video generation.

## For Beginners

Think of AnimateDiff as "teaching an image generator to make videos."

How it works:

1. Start with a text-to-image model (like Stable Diffusion)
2. Add special "motion modules" between the layers
3. These modules learn how things move in videos
4. The original image quality is preserved while adding motion

Key advantages:

- Works with any Stable Diffusion model/checkpoint
- Can use existing LoRAs, ControlNets, etc.
- Flexible: text-to-video, image-to-video, or both
- Lower training requirements than full video models

Example use cases:

- Generate a short animation from a text prompt
- Animate a still image with natural motion
- Create consistent character animations
- Style transfer for videos using SD checkpoints

## How It Works

AnimateDiff extends Stable Diffusion with motion modules that enable temporal consistency
in video generation. Unlike SVD which is trained end-to-end for video, AnimateDiff adds
motion modules to existing text-to-image models, making it highly flexible.

Architecture overview:

- Base: Standard Stable Diffusion U-Net
- Motion Modules: Temporal attention layers inserted after spatial attention
- VAE: Standard SD VAE (per-frame encoding/decoding)
- Optional: LoRA adapters for style customization

Supported modes:

- Text-to-Video: Generate video from text prompt
- Image-to-Video: Animate an input image with text guidance
- Video-to-Video: Style transfer or modify existing video

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnimateDiffModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,MotionModuleConfig,Int32,Int32)` | Initializes a new instance of AnimateDiffModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` | Gets the conditioning module. |
| `ContextLength` | Gets or sets the context length for temporal attention. |
| `ContextOverlap` | Gets or sets the context overlap for sliding window generation. |
| `LatentChannels` | Gets the number of latent channels. |
| `MotionConfig` | Gets the motion module configuration. |
| `NoisePredictor` | Gets the noise predictor. |
| `ParameterCount` | Gets the total parameter count. |
| `SupportsImageToVideo` | Gets whether image-to-video is supported. |
| `SupportsTextToVideo` | Gets whether text-to-video is supported. |
| `SupportsVideoToVideo` | Gets whether video-to-video is supported. |
| `VAE` | Gets the VAE for frame encoding/decoding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `ApplyMotionModuleWeight(Int32,Int32)` | Applies motion module temporal weighting. |
| `ApplyTemporalSmoothing(Tensor<>,Int32)` | Applies temporal smoothing across predicted noise. |
| `BlendWindowNoise(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Blends window noise into the accumulator with linear weights. |
| `Clone` | Clones this AnimateDiff model. |
| `DecodeVideoLatents(Tensor<>)` | Decodes video latents to frames. |
| `DeepCopy` | Creates a deep copy. |
| `ExtractFrameWindow(Tensor<>,Int32,Int32)` | Extracts a window of frames from video latents. |
| `GenerateFromImage(Tensor<>,Nullable<Int32>,Nullable<Int32>,Int32,Nullable<Int32>,Double,Nullable<Int32>)` | Generates video from an input image. |
| `GenerateFromText(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Double,Nullable<Int32>)` | Generates video from text using AnimateDiff. |
| `GenerateVideoWindow(Int32,Int32,Int32,Tensor<>,Tensor<>,Double,Int32,Nullable<Int32>)` | Generates a single window of video frames. |
| `GenerateWithSlidingWindow(Int32,Int32,Int32,Tensor<>,Tensor<>,Double,Int32,Nullable<Int32>)` | Generates video using sliding window approach for longer sequences. |
| `GetParameters` | Gets all parameters. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>)` | Initializes the model layers, using provided components or creating defaults. |
| `NormalizeAccumulatedNoise(Tensor<>,Tensor<>)` | Normalizes accumulated noise by weights. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts video noise for image-to-video generation. |
| `PredictWithMotionModules(Tensor<>,Int32,Tensor<>)` | Predicts noise using the U-Net with motion module awareness. |
| `SetParameters(Vector<>)` | Sets all parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `ANIMATEDIFF_LATENT_CHANNELS` | Standard AnimateDiff latent channels. |
| `DefaultHeight` | Default AnimateDiff height (SD compatible). |
| `DefaultWidth` | Default AnimateDiff width (SD compatible). |
| `LATENT_SCALE` | Standard latent scale factor. |
| `_conditioner` | Optional conditioning module for text guidance. |
| `_motionConfig` | The motion module weights/state. |
| `_unet` | The U-Net noise predictor with motion modules. |
| `_vae` | The standard VAE for frame encoding/decoding. |

