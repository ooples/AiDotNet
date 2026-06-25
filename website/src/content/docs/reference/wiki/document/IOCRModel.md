---
title: "IOCRModel<T>"
description: "Interface for OCR (Optical Character Recognition) models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for OCR (Optical Character Recognition) models.

## For Beginners

OCR is like teaching a computer to read. Given an image of text,
the model outputs the actual text content and where each word/character is located.

Example usage:

## How It Works

OCR models convert images containing text into machine-readable text strings,
along with position information for each text element.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOCRFree` | Gets whether this is an OCR-free model (end-to-end pixel-to-text). |
| `SupportedLanguages` | Gets the languages supported by this OCR model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RecognizeText(Tensor<>)` | Performs full OCR on a document image. |
| `RecognizeTextInRegion(Tensor<>,Vector<>)` | Performs OCR on a specific region of the document. |

