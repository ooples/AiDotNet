---
title: "FrameDropout<T>"
description: "Randomly drops frames from video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Video`

Randomly drops frames from video.

## For Beginners

Frame dropout randomly removes some frames from the video,
simulating frame drops in real-world video capture or network streaming.
This helps models become robust to missing frames.

## How It Works

**When to use:**

- Training for real-world video processing with potential frame drops
- Reducing overfitting to exact frame sequences
- Simulating lower frame rate videos

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrameDropout(Double,Int32,Double,Double)` | Creates a new frame dropout augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets the probability of dropping each frame. |
| `MinFramesToKeep` | Gets the minimum number of frames to keep. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>[],AugmentationContext<>)` |  |
| `GetParameters` |  |

