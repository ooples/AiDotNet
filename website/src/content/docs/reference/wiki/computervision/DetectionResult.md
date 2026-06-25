---
title: "DetectionResult<T>"
description: "Contains the results of object detection on an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection`

Contains the results of object detection on an image.

## For Beginners

This class holds all the objects detected in an image.
Each detection includes:

- A bounding box showing where the object is
- The class/category of the object (e.g., "person", "car")
- A confidence score indicating how sure the model is

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of detections. |
| `Detections` | List of all detected objects in the image. |
| `ImageHeight` | Original image height in pixels. |
| `ImageWidth` | Original image width in pixels. |
| `InferenceTime` | Time taken to run inference. |
| `ModelName` | Model name that produced these detections. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FilterByClass(Int32[])` | Filters detections by class ID. |
| `FilterByConfidence(Double)` | Filters detections by confidence threshold. |
| `SortByConfidence` | Gets detections sorted by confidence (highest first). |
| `TopN(Int32)` | Gets the top N detections by confidence. |

