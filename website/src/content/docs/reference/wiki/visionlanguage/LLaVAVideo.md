---
title: "LLaVAVideo<T>"
description: "LLaVA-Video: synthetic dataset-trained video instruction model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

LLaVA-Video: synthetic dataset-trained video instruction model.

## For Beginners

LLaVA-Video is a video-language model trained with synthetic
data for video instruction following. Default values follow the original paper settings.

## How It Works

LLaVA-Video (ByteDance, 2024) is a video instruction-tuned model trained with synthetic
video-text datasets. It uses GPT-4V-generated video descriptions as training data to build
robust video understanding capabilities, handling video question answering, temporal
reasoning, and video captioning with high-quality instruction following.

**References:**

- Paper: "Video Instruction Tuning With Synthetic Data" (ByteDance, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using LLaVA-Video's instruction-tuned visual-text fusion. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using LLaVA-Video's temporal token pooling with cross-frame attention weighting. |

