---
title: "MPLUGDocOwl2<T>"
description: "mPLUG-DocOwl 2: high-res compressing for multi-page document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

mPLUG-DocOwl 2: high-res compressing for multi-page document understanding.

## For Beginners

mPLUG-DocOwl 2 is a document understanding model for multi-page
documents with high-resolution visual compression. Default values follow the original paper
settings.

## How It Works

mPLUG-DocOwl 2 (Alibaba, 2024) handles multi-page document understanding with high-resolution
visual token compression. It processes each page at high resolution and compresses the visual
tokens to manageable lengths, enabling the model to handle multi-page PDFs and long documents
while preserving fine text details and layout structure.

**References:**

- Paper: "mPLUG-DocOwl 2: High-resolution Compressing for OCR-free Multi-page Document Understanding" (Alibaba, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using mPLUG-DocOwl 2's high-resolution compression. |

