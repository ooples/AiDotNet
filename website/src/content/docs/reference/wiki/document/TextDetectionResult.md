---
title: "TextDetectionResult<T>"
description: "Represents the result of text detection in an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of text detection in an image.

## For Beginners

Text detection finds where text is located in an image.
This result contains the bounding boxes (locations) of all detected text regions,
but not the actual text content - that requires a text recognizer.

## Properties

| Property | Summary |
|:-----|:--------|
| `BinaryMap` | Gets the binary map (final segmentation result). |
| `ProbabilityMap` | Gets the probability map showing text likelihood at each pixel. |
| `ProcessingTimeMs` | Gets the processing time in milliseconds. |
| `RegionCount` | Gets the number of detected text regions. |
| `TextRegions` | Gets the detected text regions. |
| `ThresholdMap` | Gets the threshold map (for models like DBNet that use adaptive thresholding). |

