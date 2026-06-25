---
title: "DocumentReader<T>"
description: "Document reader for OCR with layout analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.OCR.EndToEnd`

Document reader for OCR with layout analysis.

## For Beginners

DocumentReader is optimized for reading structured documents
like scanned papers, forms, and PDFs. Unlike scene text, documents have regular layouts
with clear reading order. This reader analyzes the document structure and extracts
text in logical reading order.

## How It Works

Key features:

- Layout analysis for document structure understanding
- Reading order detection
- Paragraph and line grouping
- Handles multi-column layouts
- Optimized for clean text on uniform backgrounds

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocumentReader(OCROptions<>)` | Creates a new document reader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Name` | Name of this document reader. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `GetParameterCount` | Gets the total parameter count. |
| `GetParameters` |  |
| `Predict(Tensor<>)` |  |
| `ReadDocument(Tensor<>)` | Reads a document image and returns structured text. |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |

