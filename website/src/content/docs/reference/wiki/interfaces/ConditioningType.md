---
title: "ConditioningType"
description: "Types of conditioning supported by diffusion models."
section: "API Reference"
---

`Enums` · `AiDotNet.Interfaces`

Types of conditioning supported by diffusion models.

## For Beginners

This describes what kind of input you're using to guide generation:

- Text: Written descriptions like "a beautiful sunset"
- Image: Reference images for style or content
- Audio: Sound clips for audio generation
- Control: Structural guidance like edges, poses, or depth maps
- Class: Simple class labels like "dog" or "car"

## Fields

| Field | Summary |
|:-----|:--------|
| `Audio` | Audio conditioning (e.g., audio spectrograms). |
| `Class` | Class label conditioning (e.g., ImageNet class embeddings). |
| `Control` | Spatial control conditioning (e.g., ControlNet, T2I-Adapter). |
| `Image` | Image conditioning (e.g., CLIP vision encoder, IP-Adapter). |
| `MultiModal` | Multi-modal conditioning (combines multiple types). |
| `Text` | Text conditioning (e.g., CLIP, T5 text encoders). |

