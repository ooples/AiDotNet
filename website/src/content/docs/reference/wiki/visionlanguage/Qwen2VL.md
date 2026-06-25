---
title: "Qwen2VL<T>"
description: "Qwen2-VL: Naive Dynamic Resolution with M-RoPE for multimodal positional encoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Qwen2-VL: Naive Dynamic Resolution with M-RoPE for multimodal positional encoding.

## For Beginners

Qwen2-VL introduces two key innovations. First, "Naive Dynamic
Resolution" lets it process images at any resolution without padding or cropping — the model
adapts to each image's natural size. Second, Multimodal RoPE (M-RoPE) extends positional
encoding across three dimensions: text position, spatial location in images, and temporal
position in videos. This means the model naturally understands not just what tokens say,
but where they are in text, where they are in an image, and when they appear in a video.
This unified positional encoding enables native video understanding alongside image
comprehension. Default values follow the original paper settings.

## How It Works

Qwen2-VL (Wang et al., 2024) introduces Naive Dynamic Resolution to process images at any
resolution without padding or cropping. Multimodal RoPE (M-RoPE) extends positional encoding
across text, spatial, and temporal dimensions, enabling native video understanding alongside
image comprehension.

**References:**

- Paper: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" (Wang et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Qwen2-VL's dynamic-resolution + M-RoPE architecture. |
| `GetExtraTrainableLayers` |  |

