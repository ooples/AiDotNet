---
title: "GOTOCR2<T>"
description: "GOT-OCR2: 580M unified OCR model for text, tables, charts, equations, and music scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

GOT-OCR2: 580M unified OCR model for text, tables, charts, equations, and music scores.

## For Beginners

GOT-OCR2 is a unified OCR model that handles text, tables,
charts, equations, and music scores. Default values follow the original paper settings.

## How It Works

GOT-OCR2 (StepFun, 2024) is a 580M unified end-to-end OCR model that handles diverse visual
text including plain text, tables, charts, mathematical equations, and music scores. It uses
a vision encoder with a lightweight language decoder to directly generate structured text
output from document images without separate detection and recognition stages.

**References:**

- Paper: "General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model" (StepFun, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using GOT-OCR2's unified multi-type OCR pipeline. |

