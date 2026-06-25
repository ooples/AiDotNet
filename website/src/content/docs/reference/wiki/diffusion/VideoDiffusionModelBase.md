---
title: "VideoDiffusionModelBase<T>"
description: "Base class for video diffusion models that generate temporal sequences."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion`

Base class for video diffusion models that generate temporal sequences.

## For Beginners

This is the foundation for video generation models like Stable Video Diffusion
and AnimateDiff. It extends latent diffusion to handle the temporal dimension, generating
coherent video sequences where frames are consistent over time.

## How It Works

This abstract base class provides common functionality for all video diffusion models,
including image-to-video generation, text-to-video generation, video-to-video transformation,
and frame interpolation.

Key capabilities:

- Image-to-Video: Animate a still image
- Text-to-Video: Generate video from text description
- Video-to-Video: Transform existing video style/content
- Frame interpolation: Increase frame rate smoothly

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoDiffusionModelBase(DiffusionModelOptions<>,INoiseScheduler<>,Int32,Int32,NeuralNetworkArchitecture<>)` | Initializes a new instance of the VideoDiffusionModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultFPS` |  |
| `DefaultNumFrames` |  |
| `MotionBucketId` |  |
| `NoiseAugStrength` | Gets the noise augmentation strength for input images. |
| `SupportsImageToVideo` |  |
| `SupportsTextToVideo` |  |
| `SupportsVideoToVideo` |  |
| `TemporalVAE` | Gets the temporal VAE for video encoding/decoding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoiseToVideoLatents(Tensor<>,Int32,Random)` | Adds noise to video latents at a specific timestep. |
| `ApplyGuidanceVideo(Tensor<>,Tensor<>,Double)` | Applies classifier-free guidance to video noise predictions. |
| `CreateMotionEmbedding(Int32,Int32)` | Creates a motion embedding from the motion bucket ID and FPS. |
| `DecodeVideoLatents(Tensor<>)` | Decodes video latents to frames. |
| `EncodeConditioningImage(Tensor<>,Double,Nullable<Int32>)` | Encodes a conditioning image for image-to-video generation. |
| `EncodeVideoToLatent(Tensor<>)` | Encodes a video to latent space. |
| `ExtractFrame(Tensor<>,Int32)` |  |
| `ExtractFrameLatent(Tensor<>,Int32)` | Extracts a single frame's latent from video latents. |
| `FramesToVideo(Tensor<>[])` |  |
| `GenerateFromImage(Tensor<>,Nullable<Int32>,Nullable<Int32>,Int32,Nullable<Int32>,Double,Nullable<Int32>)` |  |
| `GenerateFromText(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Double,Nullable<Int32>)` |  |
| `InsertFrameLatent(Tensor<>,Tensor<>,Int32)` | Inserts a frame latent into video latents at the specified index. |
| `InterpolateFrames(Tensor<>,Int32,FrameInterpolationMethod)` |  |
| `InterpolateFramesBlend(Tensor<>,Int32)` | Interpolates frames using blend method. |
| `InterpolateFramesDiffusion(Tensor<>,Int32)` | Interpolates frames using diffusion-based method. |
| `InterpolateFramesLinear(Tensor<>,Int32)` | Interpolates frames using linear interpolation. |
| `InterpolateFramesOpticalFlow(Tensor<>,Int32)` | Interpolates frames using optical flow (simplified). |
| `LinearBlend(Tensor<>,Tensor<>,Double)` | Linearly blends two frames. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts noise for video frames conditioned on image and motion. |
| `PredictVideoNoiseWithText(Tensor<>,Int32,Tensor<>)` | Predicts noise for video frames conditioned on text. |
| `SchedulerStepVideo(Tensor<>,Tensor<>,Int32)` | Performs a scheduler step for video latents. |
| `SetMotionBucketId(Int32)` |  |
| `VideoToVideo(Tensor<>,String,String,Double,Int32,Double,Nullable<Int32>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultFPS` | Default frames per second. |
| `_defaultNumFrames` | Default number of frames to generate. |
| `_motionBucketId` | The motion bucket ID for controlling motion intensity. |

