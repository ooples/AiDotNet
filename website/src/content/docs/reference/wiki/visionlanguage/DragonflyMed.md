---
title: "DragonflyMed<T>"
description: "Dragonfly-Med: medical image understanding model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Medical`

Dragonfly-Med: medical image understanding model.

## For Beginners

Dragonfly-Med is a vision-language model specialized for medical
image understanding with multi-resolution visual encoding. Default values follow the original
paper settings.

## How It Works

Dragonfly-Med (Together.ai, 2024) is a medical image understanding model that uses multi-resolution
visual encoding to capture fine-grained clinical details at multiple scales. It processes medical
images through a hierarchical vision encoder that handles both global context and localized
pathological features, enabling accurate diagnosis support and medical visual question answering.

**References:**

- Paper: "Dragonfly-Med: Multi-Resolution Visual Encoding for Medical Image Understanding (Together.ai, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a medical image using Dragonfly-Med's multi-resolution encoding pipeline. |

