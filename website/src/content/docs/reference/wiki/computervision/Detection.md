---
title: "Detection<T>"
description: "Represents a single detected object."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection`

Represents a single detected object.

## For Beginners

This represents one object found in the image.
It tells you:

- Where the object is (bounding box)
- What type of object it is (class ID and name)
- How confident the model is (0.0 to 1.0)
- Optionally, a pixel mask for instance segmentation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Detection` | Creates a new detection with default numeric operations. |
| `Detection(BoundingBox<>,Int32,,String)` | Creates a new detection with specified values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Area` | Gets the area of the bounding box. |
| `Box` | The bounding box defining the object's location. |
| `CenterX` | Gets the center X coordinate of the bounding box. |
| `CenterY` | Gets the center Y coordinate of the bounding box. |
| `ClassId` | The class ID (index) of the detected object. |
| `ClassName` | The human-readable name of the class (e.g., "person", "car"). |
| `Confidence` | Confidence score from 0 to 1 indicating detection certainty. |
| `Keypoints` | Optional keypoints for pose estimation. |
| `Mask` | Optional instance segmentation mask for this object. |
| `TrackId` | Optional track ID for object tracking across frames. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Gets a string representation of this detection. |

