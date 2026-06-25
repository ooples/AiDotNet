---
title: "OCRResult<T>"
description: "Represents the result of OCR (Optical Character Recognition) processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of OCR (Optical Character Recognition) processing.

## For Beginners

OCR converts images of text into machine-readable text.
This result class contains the recognized text along with position information
for each word, line, and block of text.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageConfidence` | Gets the average confidence across all recognized text. |
| `Blocks` | Gets the text blocks (paragraphs or regions). |
| `DetectedLanguage` | Gets the detected language code (ISO 639-1, e.g., "en", "zh"). |
| `FullText` | Gets the full recognized text from the document. |
| `Lines` | Gets the recognized lines of text. |
| `ProcessingTimeMs` | Gets processing time in milliseconds. |
| `RequiresDeskewing` | Gets whether the document appears to be rotated or skewed. |
| `RotationAngle` | Gets the detected rotation angle in degrees if applicable. |
| `Words` | Gets the recognized words with their positions and confidence. |

