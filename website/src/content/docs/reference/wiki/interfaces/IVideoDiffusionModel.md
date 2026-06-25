---
title: "IVideoDiffusionModel<T>"
description: "Interface for video diffusion models that generate temporal sequences."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for video diffusion models that generate temporal sequences.

## For Beginners

Video diffusion is like image diffusion, but it creates videos
instead of single images. The main challenge is making the frames look consistent
over time (no flickering or teleporting objects).

How video diffusion works:

1. The model generates multiple frames at once (typically 14-25 frames)
2. Special "temporal attention" ensures frames are consistent
3. The model can be conditioned on a starting image, text, or both

Common approaches:

- Image-to-Video (SVD): Start from an image, generate motion
- Text-to-Video (VideoCrafter): Generate video from text description
- Video-to-Video: Transform existing video with new style/content

Key challenges solved by these models:

- Temporal consistency (no flickering)
- Motion coherence (objects move naturally)
- Long-range dependencies (beginning and end are related)

## How It Works

Video diffusion models extend image diffusion to handle the temporal dimension,
generating coherent video sequences. They model both spatial (within-frame) and
temporal (across-frame) dependencies.

This interface extends `IDiffusionModel` with video-specific operations.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultFPS` | Gets the default frames per second for generated videos. |
| `DefaultNumFrames` | Gets the default number of frames generated. |
| `MotionBucketId` | Gets the motion bucket ID for controlling motion intensity (SVD-specific). |
| `SupportsImageToVideo` | Gets whether this model supports image-to-video generation. |
| `SupportsTextToVideo` | Gets whether this model supports text-to-video generation. |
| `SupportsVideoToVideo` | Gets whether this model supports video-to-video transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractFrame(Tensor<>,Int32)` | Extracts a frame from the video tensor. |
| `FramesToVideo(Tensor<>[])` | Concatenates frames into a video tensor. |
| `GenerateFromImage(Tensor<>,Nullable<Int32>,Nullable<Int32>,Int32,Nullable<Int32>,Double,Nullable<Int32>)` | Generates a video from a conditioning image. |
| `GenerateFromText(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Double,Nullable<Int32>)` | Generates a video from a text prompt. |
| `InterpolateFrames(Tensor<>,Int32,FrameInterpolationMethod)` | Interpolates between frames to increase frame rate. |
| `SetMotionBucketId(Int32)` | Sets the motion intensity for generation. |
| `VideoToVideo(Tensor<>,String,String,Double,Int32,Double,Nullable<Int32>)` | Transforms an existing video. |

