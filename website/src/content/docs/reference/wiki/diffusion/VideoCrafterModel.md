---
title: "VideoCrafterModel<T>"
description: "VideoCrafter model for high-quality text-to-video and image-to-video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

VideoCrafter model for high-quality text-to-video and image-to-video generation.

## For Beginners

VideoCrafter is like having two video generation modes in one:

Mode 1 - Text-to-Video:

- Input: "A rocket launching into space"
- Output: 5-second video of a rocket launch

Mode 2 - Image-to-Video:

- Input: Photo of a rocket on launch pad
- Output: 5-second video of the rocket launching

Key advantages:

- High visual quality (up to 1024x576 resolution)
- Long video generation (up to 16+ seconds)
- Good temporal coherence (smooth motion)
- Dual conditioning (text + image together)

Unlike AnimateDiff which adds motion to SD models, VideoCrafter is trained
end-to-end specifically for video generation, resulting in better quality.

## How It Works

VideoCrafter is a video generation model that combines the strengths of text-to-video
and image-to-video generation. It uses a dual-conditioning approach that enables
both modalities while maintaining high visual quality and temporal coherence.

Architecture:

- 3D U-Net with factorized spatial-temporal attention
- Dual cross-attention for text and image conditioning
- Temporal VAE for consistent frame encoding
- DDIM scheduler for fast inference

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoCrafterModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of VideoCrafterModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` | Gets the primary conditioning module (text). |
| `ImageConditioner` | Gets the image conditioning module. |
| `ImageConditioningScale` | Gets or sets the image conditioning scale. |
| `LatentChannels` | Gets the latent channels. |
| `NoisePredictor` | Gets the noise predictor. |
| `ParameterCount` | Gets the total parameter count. |
| `SupportsImageToVideo` | Gets whether image-to-video is supported. |
| `SupportsTextToVideo` | Gets whether text-to-video is supported. |
| `SupportsVideoToVideo` | Gets whether video-to-video is supported. |
| `TemporalVAE` | Gets the temporal VAE. |
| `UseDualConditioning` | Gets or sets whether to use dual conditioning (text + image together). |
| `VAE` | Gets the VAE. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Clones this model. |
| `CombineConditionings(Tensor<>,Tensor<>,Double)` | Combines image and text conditionings. |
| `CombineImageAndMotion(Tensor<>,Tensor<>)` | Combines image embedding with motion embedding for temporal conditioning. |
| `DecodeVideoLatents(Tensor<>)` | Decodes video latents using temporal VAE. |
| `DeepCopy` | Creates a deep copy. |
| `GenerateFromImage(Tensor<>,Nullable<Int32>,Nullable<Int32>,Int32,Nullable<Int32>,Double,Nullable<Int32>)` | Generates video from image with optional text guidance. |
| `GenerateFromImageAndText(Tensor<>,String,String,Nullable<Int32>,Int32,Double,Double,Nullable<Int32>)` | Generates video with dual conditioning (image + text). |
| `GenerateFromText(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Double,Nullable<Int32>)` | Generates video from text prompt. |
| `GetParameters` | Gets all parameters. |
| `InitializeLayers(VideoUNetPredictor<>,TemporalVAE<>)` | Initializes the model layers, using provided components or creating defaults. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts video noise for image-to-video. |
| `PredictVideoNoiseWithText(Tensor<>,Int32,Tensor<>)` | Predicts video noise with text conditioning. |
| `PredictWithDualConditioning(Tensor<>,Int32,Tensor<>,Tensor<>,Double)` | Predicts noise with dual conditioning. |
| `SetParameters(Vector<>)` | Sets all parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default VideoCrafter height. |
| `DefaultWidth` | Default VideoCrafter width. |
| `LATENT_CHANNELS` | VideoCrafter latent channels. |
| `LATENT_SCALE` | Standard latent scale factor. |
| `_imageConditioner` | The image conditioning module. |
| `_temporalVAE` | The temporal VAE for video encoding/decoding. |
| `_textConditioner` | The text conditioning module. |
| `_videoUNet` | The VideoUNet noise predictor with dual conditioning. |

