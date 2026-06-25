---
title: "DocumentVLMOptions"
description: "Base configuration options for document understanding vision-language models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Document`

Base configuration options for document understanding vision-language models.

## For Beginners

These options configure the Document model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocumentVLMOptions` | Initializes a new instance with default values. |
| `DocumentVLMOptions(DocumentVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOcrFree` | Gets or sets whether this model operates OCR-free (no external OCR required). |
| `MaxOutputTokens` | Gets or sets the maximum output text length in tokens. |
| `MaxPages` | Gets or sets the maximum document page count supported. |

