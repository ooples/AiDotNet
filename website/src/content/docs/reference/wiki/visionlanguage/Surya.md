---
title: "Surya<T>"
description: "Surya: multi-language OCR with layout analysis support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

Surya: multi-language OCR with layout analysis support.

## For Beginners

Surya is a multi-language OCR model with layout analysis for
text detection and recognition in 90+ languages. Default values follow the original paper
settings.

## How It Works

Surya (Datalab, 2024) is a multi-language OCR toolkit with integrated layout analysis support.
It combines text detection, text recognition, and document layout analysis in a unified
pipeline, supporting over 90 languages with line-level and word-level OCR alongside
structural layout detection for tables, figures, and headers.

**References:**

- Paper: "Surya: Multi-language OCR Toolkit" (Datalab, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using Surya's multi-language OCR pipeline. |

