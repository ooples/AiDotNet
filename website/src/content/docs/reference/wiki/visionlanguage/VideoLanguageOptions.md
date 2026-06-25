---
title: "VideoLanguageOptions"
description: "Base configuration options for video-language models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.VideoLanguage`

Base configuration options for video-language models.

## For Beginners

These options configure the VideoLanguage model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoLanguageOptions` | Initializes a new instance with default values. |
| `VideoLanguageOptions(VideoLanguageOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets or sets the language model backbone name. |
| `MaxFrames` | Gets or sets the maximum number of video frames the model can process. |
| `ProjectionDim` | Gets or sets the MLP projection hidden dimension. |
| `SystemPrompt` | Gets or sets the system prompt for chat mode. |

