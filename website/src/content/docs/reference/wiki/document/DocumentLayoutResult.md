---
title: "DocumentLayoutResult<T>"
description: "Represents the result of document layout detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of document layout detection.

## For Beginners

Layout detection identifies different parts of a document
(text blocks, tables, figures, etc.) and their locations. This result class
contains all the detected regions with their bounding boxes and types.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProcessingTimeMs` | Gets processing time in milliseconds. |
| `ReadingOrder` | Gets the reading order of text regions (if detected). |
| `Regions` | Gets the detected layout regions. |
| `TotalRegions` | Gets the total number of detected regions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetHighConfidenceRegions(Double)` | Gets regions with confidence above a threshold. |
| `GetRegionsByType(LayoutElementType)` | Gets regions filtered by element type. |

