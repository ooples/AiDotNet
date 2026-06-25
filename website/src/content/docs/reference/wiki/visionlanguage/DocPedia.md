---
title: "DocPedia<T>"
description: "DocPedia: frequency-domain document understanding model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

DocPedia: frequency-domain document understanding model.

## For Beginners

DocPedia is a document understanding model that processes documents
in the frequency domain for text and layout analysis. Default values follow the original
paper settings.

## How It Works

DocPedia (2024) unleashes the power of large multimodal models for document understanding by
operating in the frequency domain. It converts document images into frequency representations
via DCT transforms, enabling the model to capture both structural layout patterns and fine
text details without requiring explicit OCR preprocessing.

**References:**

- Paper: "DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using DocPedia's frequency-domain pipeline. |

