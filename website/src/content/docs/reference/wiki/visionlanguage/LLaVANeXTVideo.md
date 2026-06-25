---
title: "LLaVANeXTVideo<T>"
description: "LLaVA-NeXT-Video: average pooling for frame token reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

LLaVA-NeXT-Video: average pooling for frame token reduction.

## For Beginners

LLaVA-NeXT-Video is a video understanding model that extends
image LLMs to video through efficient frame token pooling. Default values follow the
original paper settings.

## How It Works

LLaVA-NeXT-Video (ByteDance, 2024) extends LLaVA-NeXT to video understanding using average
pooling for efficient frame token reduction. It processes video frames through a shared image
encoder and pools temporal tokens to reduce sequence length, enabling zero-shot video
question answering and temporal reasoning without video-specific training.

**References:**

- Paper: "LLaVA-NeXT: A Strong Zero-shot Video Understanding Model" (ByteDance, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using LLaVA-NeXT-Video's AnyRes dynamic resolution processing. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using LLaVA-NeXT-Video's AnyRes + temporal pooling. |

