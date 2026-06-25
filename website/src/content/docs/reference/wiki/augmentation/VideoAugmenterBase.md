---
title: "VideoAugmenterBase<T>"
description: "Base class for video data augmentations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation.Video`

Base class for video data augmentations.

## For Beginners

Video augmentation transforms sequences of frames to improve
model robustness to temporal variations. It combines:

- Temporal augmentations (time-based): cropping, reversing, speed changes
- Spatial augmentations (frame-based): flips, rotations, color changes applied consistently across frames

## How It Works

Video data is represented as an array of ImageTensor frames.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoAugmenterBase(Double,Double)` | Initializes a new video augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameRate` | Gets or sets the frame rate of the video in frames per second. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDuration(ImageTensor<>[])` | Gets the duration of the video in seconds. |
| `GetFrameCount(ImageTensor<>[])` | Gets the number of frames in the video. |
| `GetParameters` |  |
| `ValidateFrameDimensions(ImageTensor<>[])` | Validates that all frames have the same dimensions. |

