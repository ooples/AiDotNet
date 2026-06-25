---
title: "VideoLLaMA3<T>"
description: "VideoLLaMA 3: frontier multimodal for image and video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

VideoLLaMA 3: frontier multimodal for image and video.

## For Beginners

VideoLLaMA 3 is Alibaba's frontier multimodal model for
advanced image and video understanding. Default values follow the original paper settings.

## How It Works

VideoLLaMA 3 (Alibaba, 2025) is a frontier multimodal foundation model for both image
and video understanding. It builds on the VideoLLaMA series with improved visual encoders,
enhanced temporal modeling, and expanded training on large-scale image-video-text datasets
for state-of-the-art performance on video comprehension benchmarks.

**References:**

- Paper: "VideoLLaMA 3: Frontier Multimodal Foundation Models" (Alibaba, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using VideoLLaMA 3's any-resolution visual tokenizer with adaptive spatial token merging. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using VideoLLaMA 3's adaptive spatial-temporal token merging. |

