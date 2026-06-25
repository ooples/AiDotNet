---
title: "TextMonkey<T>"
description: "TextMonkey: OCR-free text understanding with shifted window attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

TextMonkey: OCR-free text understanding with shifted window attention.

## For Beginners

TextMonkey is an OCR-free document understanding model with
shifted window attention for efficient text processing. Default values follow the original
paper settings.

## How It Works

TextMonkey (HUST, 2024) is an OCR-free large multimodal model for document understanding
that uses shifted window attention to efficiently process high-resolution document images.
The shifted window mechanism reduces computational cost while maintaining fine-grained
text recognition ability, enabling direct document comprehension without separate OCR stages.

**References:**

- Paper: "TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document" (HUST, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using TextMonkey's shifted window attention. |

