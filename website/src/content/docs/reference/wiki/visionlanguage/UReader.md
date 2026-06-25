---
title: "UReader<T>"
description: "UReader: universal OCR-free visually-situated language model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

UReader: universal OCR-free visually-situated language model.

## For Beginners

UReader is a universal OCR-free model for understanding text
in documents, web pages, and natural scenes. Default values follow the original paper
settings.

## How It Works

UReader (2024) is a universal OCR-free model for visually-situated language understanding.
It handles diverse document types including scanned documents, web pages, slides, and natural
scene text through adaptive resolution processing and shape-adaptive cropping, unifying
document understanding across different visual text formats without requiring OCR preprocessing.

**References:**

- Paper: "UReader: Universal OCR-free Visually-situated Language Understanding" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using UReader's shape-adaptive cropping pipeline. |

