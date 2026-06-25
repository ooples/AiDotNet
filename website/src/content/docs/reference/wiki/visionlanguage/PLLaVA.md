---
title: "PLLaVA<T>"
description: "PLLaVA: parameter-free pooling extension from images to video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

PLLaVA: parameter-free pooling extension from images to video.

## For Beginners

PLLaVA is a video-language model that extends image LLMs to
video using parameter-free pooling. Default values follow the original paper settings.

## How It Works

PLLaVA (HKU, 2024) extends image-based LLaVA models to video through parameter-free pooling
operations. Without adding any learnable parameters, it aggregates frame-level visual tokens
using average pooling along the temporal dimension, providing a simple yet effective baseline
for video dense captioning and video question answering.

**References:**

- Paper: "PLLaVA: Parameter-free LLaVA Extension from Images to Videos" (HKU, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using PLLaVA's parameter-free approach. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using PLLaVA's parameter-free adaptive pooling. |

