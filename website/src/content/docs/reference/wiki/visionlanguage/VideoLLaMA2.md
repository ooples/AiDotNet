---
title: "VideoLLaMA2<T>"
description: "VideoLLaMA 2: spatial-temporal convolution for video tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

VideoLLaMA 2: spatial-temporal convolution for video tokens.

## For Beginners

VideoLLaMA 2 is a video-language model with spatial-temporal
convolution for efficient video token processing. Default values follow the original
paper settings.

## How It Works

VideoLLaMA 2 (Alibaba, 2024) advances spatial-temporal modeling for video understanding
using convolution-based video token aggregation. It applies spatial-temporal convolutions
to compress frame-level visual tokens along both spatial and temporal dimensions, with
optional audio branch support for multi-modal video understanding.

**References:**

- Paper: "VideoLLaMA 2: Advancing Spatial-Temporal Modeling" (Alibaba, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using VideoLLaMA 2's STC connector in single-frame mode. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using VideoLLaMA 2's Spatial-Temporal Convolution (STC) connector. |

