---
title: "UnifiedVisionOptions"
description: "Base configuration options for unified understanding + generation vision models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Unified`

Base configuration options for unified understanding + generation vision models.

## For Beginners

These options configure the UnifiedVision model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UnifiedVisionOptions` | Initializes a new instance with default values. |
| `UnifiedVisionOptions(UnifiedVisionOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets or sets the language model backbone name. |
| `NumVisualTokens` | Gets or sets the number of discrete visual tokens in the vocabulary. |
| `OutputImageSize` | Gets or sets the generated image resolution. |
| `SupportsGeneration` | Gets or sets whether the model supports image generation. |

