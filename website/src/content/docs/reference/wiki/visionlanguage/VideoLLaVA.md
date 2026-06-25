---
title: "VideoLLaVA<T>"
description: "Video-LLaVA: united visual representation for video understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

Video-LLaVA: united visual representation for video understanding.

## For Beginners

Video-LLaVA is a video-language model that aligns image and
video features for unified visual understanding. Default values follow the original paper
settings.

## How It Works

Video-LLaVA (PKU, 2024) learns united visual representations for video understanding by
aligning image and video features before projection into the language model. It processes
both images and videos through separate encoders that are aligned into a shared feature
space, enabling the model to leverage image-text knowledge for video understanding.

**References:**

- Paper: "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection" (PKU, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using Video-LLaVA's alignment-before-projection approach. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using Video-LLaVA's unified alignment before projection. |

