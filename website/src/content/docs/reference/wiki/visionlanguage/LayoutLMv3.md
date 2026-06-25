---
title: "LayoutLMv3<T>"
description: "LayoutLMv3: unified text, image, and layout pre-training for document AI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

LayoutLMv3: unified text, image, and layout pre-training for document AI.

## For Beginners

LayoutLMv3 is a document AI model from Microsoft that jointly
understands text, images, and layout positions. Default values follow the original paper
settings.

## How It Works

LayoutLMv3 (Microsoft, 2022) unifies text, image, and layout pre-training for document AI.
It uses unified text and image masking objectives to learn cross-modal representations
where text tokens, visual patches, and 2D layout positions are jointly encoded in a single
transformer, enabling document classification, information extraction, and layout analysis.

**References:**

- Paper: "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (Microsoft, 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using LayoutLMv3's unified multimodal pipeline. |

