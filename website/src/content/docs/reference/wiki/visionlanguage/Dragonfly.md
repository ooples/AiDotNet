---
title: "Dragonfly<T>"
description: "Dragonfly: multi-resolution visual encoding VLM for fine-grained understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Dragonfly: multi-resolution visual encoding VLM for fine-grained understanding.

## For Beginners

Dragonfly is a vision-language model that looks at images at
multiple zoom levels — like how you might first glance at a whole photo, then zoom into
interesting parts for more detail. Its multi-resolution zoom module automatically identifies
important regions and processes them at higher resolution while keeping the overall scene
context at lower resolution. This approach is more efficient than processing the entire
image at high resolution, while still capturing fine details where they matter. Default
values follow the original paper settings.

## How It Works

Dragonfly (Together AI, 2024) introduces multi-resolution visual encoding that processes images
at multiple zoom levels simultaneously. It uses a "zoom-in" module that identifies important
regions of the image and processes them at higher resolution, while maintaining a global view
at lower resolution. This multi-resolution approach gives the model both broad scene understanding
and fine-grained detail recognition without requiring extremely high-resolution processing of
the entire image.

**References:**

- Paper: "Dragonfly: Multi-Resolution Zoom Supercharges Large Visual-Language Model" (Together AI, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Dragonfly's multi-resolution visual encoding with zoom-and-select. |

