---
title: "Pix2Struct<T>"
description: "Pix2Struct: screenshot parsing pre-training for visual language understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

Pix2Struct: screenshot parsing pre-training for visual language understanding.

## For Beginners

Pix2Struct is a document model from Google pre-trained on
screenshots for visual language understanding. Default values follow the original paper
settings.

## How It Works

Pix2Struct (Google, 2023) uses screenshot parsing as pre-training for visual language
understanding. Pre-trained on web page screenshots with their underlying HTML/DOM structure,
it learns to map visual layouts to structured text representations, excelling at chart
understanding, infographic parsing, and UI interpretation tasks.

**References:**

- Paper: "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding" (Google, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates structured text from a document/screenshot using Pix2Struct's variable-resolution pipeline. |

