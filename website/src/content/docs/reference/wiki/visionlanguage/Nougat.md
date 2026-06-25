---
title: "Nougat<T>"
description: "Nougat: neural OCR for academic documents converting PDF to Markdown."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

Nougat: neural OCR for academic documents converting PDF to Markdown.

## For Beginners

Nougat is a document model from Meta that converts academic
PDFs to Markdown with equations and tables preserved. Default values follow the original
paper settings.

## How It Works

Nougat (Meta, 2023) is a neural OCR model for academic documents that converts PDF pages
directly to Markdown text. It uses a Swin Transformer encoder to process document page images
and an autoregressive text decoder to generate structured Markdown output, preserving
mathematical equations, tables, and formatting from scientific papers.

**References:**

- Paper: "Nougat: Neural Optical Understanding for Academic Documents" (Meta, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates Markdown text from an academic document image using Nougat's pipeline. |

