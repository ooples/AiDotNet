---
title: "IDocumentUnderstandingModel<T>"
description: "Interface for document understanding models that process text-heavy images, documents, charts, and tables."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for document understanding models that process text-heavy images, documents, charts, and tables.

## How It Works

Document understanding models specialize in extracting and reasoning about textual content
from document images, including OCR, layout analysis, table extraction, and document QA.
Architectures include:

- LayoutLM: Multimodal pre-training with text, layout, and image
- Donut/Nougat: OCR-free document understanding via image-to-text generation
- Pix2Struct: Screenshot parsing pre-training for visual language understanding
- mPLUG-DocOwl: Modular document understanding with visual abstractor

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOcrFree` | Gets whether this model supports OCR-free document understanding (no external OCR needed). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerDocumentQuestion(Tensor<>,String)` | Answers a question about a document image. |
| `ExtractText(Tensor<>)` | Extracts text content from a document image. |

