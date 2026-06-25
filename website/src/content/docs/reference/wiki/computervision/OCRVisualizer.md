---
title: "OCRVisualizer<T>"
description: "Visualizes OCR (Optical Character Recognition) results on images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Visualization`

Visualizes OCR (Optical Character Recognition) results on images.

## For Beginners

This class draws text region bounding boxes, polygons,
and recognized text on images to visualize OCR results.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OCRVisualizer(VisualizationOptions)` | Creates a new OCR visualizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateTextOverlay(Tensor<>,OCRResult<>)` | Creates a text overlay image with recognized text shown at original positions. |
| `Visualize(Tensor<>,OCRResult<>)` | Draws OCR results on an image. |
| `VisualizeBoxesOnly(Tensor<>,OCRResult<>)` | Draws only text bounding boxes without text labels. |
| `VisualizeDocumentLayout(Tensor<>,DocumentLayoutResult<>)` | Visualizes document layout analysis results. |

