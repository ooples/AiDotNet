---
title: "KimiVLOptions"
description: "Configuration options for Kimi-VL: MoE VLM with MoonViT and long-context processing."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Reasoning`

Configuration options for Kimi-VL: MoE VLM with MoonViT and long-context processing.

## For Beginners

These options configure the KimiVL model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KimiVLOptions(KimiVLOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActiveParameters` | Gets or sets the active parameter count in billions (MoE routing). |
| `EnableLongContext` | Gets or sets whether to enable 128K long-context mode. |
| `TotalParameters` | Gets or sets the total parameter count in billions. |

