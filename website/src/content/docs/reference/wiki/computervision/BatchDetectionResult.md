---
title: "BatchDetectionResult<T>"
description: "Represents a batch of detection results for multiple images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection`

Represents a batch of detection results for multiple images.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageInferenceTime` | Gets the average inference time per image. |
| `BatchSize` | Gets the batch size. |
| `Item(Int32)` | Gets the result for a specific image index. |
| `Results` | Detection results for each image in the batch. |
| `TotalDetections` | Gets the total number of detections across all images. |
| `TotalInferenceTime` | Total time to process the entire batch. |

