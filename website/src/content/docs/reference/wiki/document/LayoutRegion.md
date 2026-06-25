---
title: "LayoutRegion<T>"
description: "Represents a single detected layout region in a document."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents a single detected layout region in a document.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBox` | Gets the bounding box as [x1, y1, x2, y2] in pixels or normalized coordinates. |
| `Confidence` | Gets the confidence score for this detection (0-1). |
| `ConfidenceLevel` | Gets the classification of confidence level. |
| `ConfidenceValue` | Gets the confidence as a double value for comparison operations. |
| `ElementType` | Gets the type of layout element. |
| `Index` | Gets the region index in the original detection order. |
| `ParentIndex` | Gets the parent region index if this is a nested element. |
| `TextContent` | Gets the text content if available (requires OCR). |

