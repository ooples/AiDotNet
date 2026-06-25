---
title: "IUnifiedVisionModel<T>"
description: "Interface for unified vision models that can both understand and generate images."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for unified vision models that can both understand and generate images.

## How It Works

Unified models combine visual understanding (captioning, VQA) and visual generation
(image synthesis) in a single architecture. Approaches include:

- Chameleon/Show-o: discrete visual tokens for unified autoregressive generation
- Janus: decoupled visual encoding for understanding vs. generation
- Transfusion: mixed autoregressive + diffusion loss in one transformer

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGeneration` | Gets whether this model supports image generation (not just understanding). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateImage(String)` | Generates an image tensor from a text description. |

