---
title: "IImageSafetyModule<T>"
description: "Interface for safety modules that operate on image content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for safety modules that operate on image content.

## For Beginners

Image safety modules check pictures and generated images for
harmful content. They can detect things like nudity, violence, manipulated photos
(deepfakes), and whether an image was created by AI.

## How It Works

Image safety modules analyze image tensors for safety risks such as NSFW content,
graphic violence, deepfakes, and AI-generated content.

**References:**

- UnsafeBench: 11 categories of unsafe images (Qu et al., 2024)
- USD: Scene-graph NSFW detection (USENIX Security 2025)
- Safe-CLIP: Removing NSFW concepts from CLIP (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateImage(Tensor<>)` | Evaluates the given image tensor for safety and returns any findings. |

