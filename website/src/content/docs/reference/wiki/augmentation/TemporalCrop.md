---
title: "TemporalCrop<T>"
description: "Randomly crops a temporal segment from video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Video`

Randomly crops a temporal segment from video.

## For Beginners

Temporal cropping takes a random consecutive portion of the video.
For example, from a 10-second video, it might extract 8 seconds starting at a random point.
This helps models learn to recognize actions regardless of when they occur in the video.

## How It Works

**When to use:**

- Action recognition where action can occur anywhere in the video
- Training with variable-length videos
- Reducing overfitting to specific temporal positions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalCrop(Double,Double,Double,Double)` | Creates a new temporal crop augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxCropRatio` | Gets the maximum crop ratio (fraction of original length to keep). |
| `MinCropRatio` | Gets the minimum crop ratio (fraction of original length to keep). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>[],AugmentationContext<>)` |  |
| `GetParameters` |  |

