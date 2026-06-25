---
title: "InternVL2<T>"
description: "InternVL2: dynamic resolution with pixel shuffle and InternLM2 backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

InternVL2: dynamic resolution with pixel shuffle and InternLM2 backbone.

## For Beginners

InternVL2 improves on InternVL by adding dynamic resolution
support — it can process images at different sizes by splitting them into tiles and using
a "pixel shuffle" technique to efficiently downsample the visual features. This means it
handles both small icons and large high-resolution images well. It switches the language
backbone to InternLM2 and achieves performance competitive with commercial models like
GPT-4V on many benchmarks, making it one of the strongest open-source multimodal models
available. Default values follow the original paper settings.

## How It Works

InternVL2 (Chen et al., 2024) introduces dynamic resolution support through pixel shuffle
downsampling and dynamic image tiling. Combined with InternLM2 as the language backbone,
it achieves competitive performance with commercial models across diverse benchmarks.

**References:**

- Paper: "InternVL2: Better than the Best - Expanding Performance Boundaries of Open-Source Multimodal Models" (Chen et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using InternVL2's pixel shuffle + dynamic resolution architecture. |

