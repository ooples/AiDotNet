---
title: "RecognizedText<T>"
description: "Represents recognized text in an image region."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.OCR`

Represents recognized text in an image region.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecognizedText(String,)` | Creates a new recognized text. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Box` | Bounding box around the text. |
| `CharacterConfidences` | Per-character confidences (if available). |
| `Confidence` | Overall confidence of the recognition. |
| `Language` | Language detected for this text. |
| `Polygon` | Polygon points for rotated/curved text. |
| `Text` | The recognized text string. |

