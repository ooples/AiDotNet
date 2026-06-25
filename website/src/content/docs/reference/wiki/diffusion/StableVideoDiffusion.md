---
title: "StableVideoDiffusion<T>"
description: "Stable Video Diffusion (SVD) model for image-to-video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Stable Video Diffusion (SVD) model for image-to-video generation.

## For Beginners

Think of SVD as "making a picture come to life."
You give it a single image, and it generates a short video showing
how that scene might animate:

Example workflow:

1. Input: Photo of a waterfall
2. SVD analyzes the scene and understands what should move
3. Output: 4-second video showing water flowing, mist rising

Key features:

- Image-to-video: Primary use case, animate still images
- Motion control: Adjust how much motion to add (motion bucket)
- Configurable length: Generate different numbers of frames
- High quality: Based on Stable Diffusion's proven architecture

Compared to text-to-video:

- More predictable results (scene is defined by input image)
- Better quality (less ambiguity than text prompts)
- Faster generation (can use fewer denoising steps)

## How It Works

Stable Video Diffusion generates short video clips from a single input image.
It extends the Stable Diffusion architecture with temporal awareness, using
a 3D U-Net for noise prediction and a temporal VAE for encoding/decoding.

Technical specifications:

- Default resolution: 576x1024 or 1024x576
- Default frames: 25 frames at 7 FPS (~3.5 seconds)
- Motion bucket ID: 1-255 (127 = moderate motion)
- Noise augmentation: 0.02 default for conditioning image
- Latent space: 4 channels, 8x spatial downsampling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableVideoDiffusion(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Double)` | Initializes a new instance of StableVideoDiffusion with default parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` | Gets the conditioning module if available. |
| `LatentChannels` | Gets the number of latent channels (4 for SVD). |
| `NoisePredictor` | Gets the noise predictor used by this model. |
| `ParameterCount` | Gets the total number of parameters in the model. |
| `SupportsImageToVideo` | Gets whether this model supports image-to-video generation. |
| `SupportsTextToVideo` | Gets whether this model supports text-to-video generation. |
| `SupportsVideoToVideo` | Gets whether this model supports video-to-video transformation. |
| `TemporalVAE` | Gets the temporal VAE specifically for video operations. |
| `VAE` | Gets the VAE used by this model for image encoding. |
| `VideoUNet` | Gets the video U-Net predictor with image conditioning support. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyEndImageGuidance(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Applies soft guidance toward end image in latent space. |
| `Clone` | Creates a clone of this StableVideoDiffusion model. |
| `CreateDefaultScheduler` | Creates the default DDIM scheduler for SVD. |
| `CreateDefaultTemporalVAE` | Creates the default TemporalVAE for SVD. |
| `CreateDefaultVideoUNet` | Creates the default VideoUNet predictor for SVD. |
| `CreateMotionEmbedding(Int32,Int32)` | Creates SVD-specific motion embedding. |
| `DecodeVideoLatents(Tensor<>)` | Decodes video latents using the temporal VAE. |
| `DeepCopy` | Creates a deep copy of this model. |
| `EncodeConditioningImage(Tensor<>,Double,Nullable<Int32>)` | Encodes a conditioning image with SVD-specific processing. |
| `GenerateFromImage(Tensor<>,Nullable<Int32>,Nullable<Int32>,Int32,Nullable<Int32>,Double,Nullable<Int32>)` | Generates a video from an input image using image-to-video diffusion. |
| `GenerateWithEndImageGuidance(Tensor<>,Tensor<>,Int32,Int32,Nullable<Int32>)` | Generates video with motion guidance from a secondary image. |
| `GenerateWithFirstFrame(Tensor<>,Int32,Int32,Int32,Nullable<Int32>)` | Generates video with explicit first frame control. |
| `GetParameters` | Gets the flattened parameters of all components. |
| `GetRecommendedResolution(Double)` | Gets the recommended resolution for SVD generation. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts noise for video frames conditioned on image and motion. |
| `ProjectConditioningValue(Span<>,Double,Int32,Int32)` | Projects a single conditioning value into an embedding using sinusoidal timestep projection. |
| `SetParameters(Vector<>)` | Sets the parameters for all components. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_NOISE_AUG_STRENGTH` | Default noise augmentation strength for conditioning image. |
| `DefaultHeight` | Default height for SVD generation. |
| `DefaultWidth` | Default width for SVD generation. |
| `SVD_LATENT_CHANNELS` | Standard SVD latent channels. |
| `SVD_LATENT_SCALE` | Standard SVD latent scale factor. |
| `_conditioner` | Optional conditioning module for text guidance. |
| `_noiseAugmentStrength` | Noise augmentation strength for micro-conditioning. |
| `_temporalVAE` | The temporal VAE for video encoding/decoding. |
| `_videoUNet` | The VideoUNet noise predictor. |

