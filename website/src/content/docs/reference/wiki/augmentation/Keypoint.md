---
title: "Keypoint<T>"
description: "Represents a keypoint annotation for pose estimation and landmark detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Represents a keypoint annotation for pose estimation and landmark detection.

## For Beginners

Think of keypoints as dots marking important locations
on an object. For a human, these might be the shoulders, elbows, wrists, etc.
When you flip or rotate an image, these points need to move accordingly.

## How It Works

Keypoints are specific points of interest on objects, typically used for:

- Human pose estimation (joints, facial landmarks)
- Animal pose estimation
- Object landmark detection (car wheels, furniture corners)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Keypoint` | Creates an empty keypoint. |
| `Keypoint(,,Int32,String,Int32)` | Creates a keypoint with the specified coordinates. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Gets or sets the confidence score (if from a detector). |
| `ImageHeight` | Gets or sets the image height (for normalized coordinates). |
| `ImageWidth` | Gets or sets the image width (for normalized coordinates). |
| `Index` | Gets or sets the keypoint index within the skeleton. |
| `IsNormalized` | Gets or sets whether coordinates are normalized to [0, 1]. |
| `Metadata` | Gets or sets additional metadata. |
| `Name` | Gets or sets the keypoint name/label. |
| `ParentIndex` | Gets or sets the parent keypoint index (for skeleton hierarchy). |
| `Visibility` | Gets or sets the visibility state. |
| `X` | Gets or sets the X coordinate. |
| `Y` | Gets or sets the Y coordinate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of this keypoint. |
| `DistanceTo(Keypoint<>)` | Calculates the Euclidean distance to another keypoint. |
| `IsLabeled` | Checks if this keypoint is labeled (visible or occluded). |
| `IsVisible` | Checks if this keypoint is visible. |
| `IsWithinBounds(Int32,Int32)` | Checks if this keypoint is within image boundaries. |
| `OKS(Keypoint<>,Double,Double)` | Calculates the Object Keypoint Similarity (OKS) score. |
| `ToAbsolute` | Converts to absolute pixel coordinates. |
| `ToNormalized` | Converts to normalized coordinates [0, 1]. |

