---
title: "SpeedChange<T>"
description: "Changes the playback speed of video by resampling frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Video`

Changes the playback speed of video by resampling frames.

## For Beginners

Speed change makes video play faster or slower by
keeping/skipping/duplicating frames. This simulates different action speeds
and helps models recognize actions regardless of how fast they're performed.

## How It Works

**Speed factors:**

- 2.0 = double speed (skip every other frame)
- 1.0 = normal speed
- 0.5 = half speed (duplicate frames)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeedChange(Double,Double,Double,Double)` | Creates a new speed change augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSpeed` | Gets the maximum speed factor. |
| `MinSpeed` | Gets the minimum speed factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>[],AugmentationContext<>)` |  |
| `GetParameters` |  |

