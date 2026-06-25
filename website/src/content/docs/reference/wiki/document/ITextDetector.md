---
title: "ITextDetector<T>"
description: "Interface for text detection models that locate text regions in images."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for text detection models that locate text regions in images.

## For Beginners

Text detection is the first step in reading text from images.
It's like highlighting all the places where text appears, but not actually reading it.
After detection, a text recognizer reads the actual characters in each highlighted region.

Example usage:

## How It Works

Text detection models find where text appears in an image without reading the text.
They output bounding boxes (polygons or rectangles) around text regions.

## Properties

| Property | Summary |
|:-----|:--------|
| `MinTextHeight` | Gets the minimum detectable text height in pixels. |
| `SupportsPolygonOutput` | Gets whether this detector outputs polygon bounding boxes (vs axis-aligned rectangles). |
| `SupportsRotatedText` | Gets whether this detector supports rotated text detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectText(Tensor<>)` | Detects text regions in an image. |
| `DetectText(Tensor<>,Double)` | Detects text regions with a custom confidence threshold. |
| `GetProbabilityMap(Tensor<>)` | Gets the probability map showing text likelihood at each pixel. |

