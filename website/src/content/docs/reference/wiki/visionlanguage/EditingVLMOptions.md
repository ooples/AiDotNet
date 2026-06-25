---
title: "EditingVLMOptions"
description: "Base configuration options for image editing VLMs."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Editing`

Base configuration options for image editing VLMs.

## For Beginners

These options configure the Editing model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EditingVLMOptions` | Initializes a new instance with default values. |
| `EditingVLMOptions(EditingVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets or sets the guidance scale for classifier-free guidance. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion denoising steps. |
| `OutputImageSize` | Gets or sets the output image resolution. |

