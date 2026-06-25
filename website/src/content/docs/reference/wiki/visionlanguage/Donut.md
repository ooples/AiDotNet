---
title: "Donut<T>"
description: "Donut: OCR-free document understanding transformer using Swin encoder + BART decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

Donut: OCR-free document understanding transformer using Swin encoder + BART decoder.

## For Beginners

Donut is a document understanding model that processes document
images directly without OCR. Default values follow the original paper settings.

## How It Works

Donut (NAVER, 2022) is an OCR-free document understanding transformer that uses a Swin
Transformer encoder to process document images and a BART decoder to generate structured
text output. It eliminates the need for explicit OCR by directly mapping pixel inputs
to text sequences for tasks like document parsing, information extraction, and VQA.

**References:**

- Paper: "OCR-free Document Understanding Transformer" (NAVER, 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using Donut's OCR-free Swin+BART pipeline. |
| `GetOptions` |  |

