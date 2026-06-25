---
title: "RemoteSensingVLMOptions"
description: "Base configuration options for remote sensing vision-language models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.RemoteSensing`

Base configuration options for remote sensing vision-language models.

## For Beginners

These options configure the RemoteSensing model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RemoteSensingVLMOptions` | Initializes a new instance with default values. |
| `RemoteSensingVLMOptions(RemoteSensingVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GroundSampleDistance` | Gets or sets the ground sample distance in meters. |
| `LanguageModelName` | Gets or sets the language model backbone name. |
| `SupportedBands` | Gets or sets the supported image bands (e.g., "RGB", "Multispectral"). |

