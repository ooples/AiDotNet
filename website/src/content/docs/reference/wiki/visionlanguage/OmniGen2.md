---
title: "OmniGen2<T>"
description: "OmniGen2: dual-path architecture with parameter decoupling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

OmniGen2: dual-path architecture with parameter decoupling.

## For Beginners

OmniGen2 is a unified model for image generation using dual-path
architecture with parameter decoupling. Default values follow the original paper settings.

## How It Works

OmniGen2 (THU, 2025) advances unified image generation with a dual-path architecture that
decouples understanding and generation parameters. One path handles visual comprehension
while the other specializes in image generation via diffusion, allowing each path to be
independently optimized while sharing a common language backbone for instruction following.

**References:**

- Paper: "OmniGen2: Advancing Unified Image Generation with Dual-Path Architecture" (THU, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using OmniGen2's understanding path. |
| `GenerateImage(String)` | Generates an image from text using OmniGen2's rectified flow generation path. |

