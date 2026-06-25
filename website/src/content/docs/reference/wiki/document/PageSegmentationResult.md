---
title: "PageSegmentationResult<T>"
description: "Represents the result of page segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of page segmentation.

## For Beginners

Page segmentation divides a document page into different
types of regions (paragraphs, figures, tables, etc.). This result contains
all detected regions with their types and locations.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassProbabilities` | Gets the class probabilities for each pixel (shape: [height, width, num_classes]). |
| `ProcessingTimeMs` | Gets the processing time in milliseconds. |
| `ReadingOrder` | Gets the reading order of text regions (indices into Regions list). |
| `RegionCount` | Gets the total number of detected regions. |
| `Regions` | Gets the detected document regions. |
| `SegmentationMask` | Gets the pixel-level segmentation mask (class index per pixel). |

