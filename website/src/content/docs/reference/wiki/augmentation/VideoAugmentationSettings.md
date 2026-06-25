---
title: "VideoAugmentationSettings"
description: "Video-specific augmentation settings with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Video-specific augmentation settings with industry-standard defaults.

## For Beginners

These settings control how video data is augmented.
Video augmentation applies both spatial (image-based) and temporal (time-based) transforms.

## Properties

| Property | Summary |
|:-----|:--------|
| `CropRatio` | Gets or sets the fraction of frames to keep. |
| `DropoutRate` | Gets or sets the dropout rate for frames. |
| `EnableFrameDropout` | Gets or sets whether frame dropout is enabled. |
| `EnableSpatialTransforms` | Gets or sets whether spatial transforms (image augmentations) are applied to frames. |
| `EnableSpeedChange` | Gets or sets whether speed change is enabled. |
| `EnableTemporalCrop` | Gets or sets whether temporal crop is enabled. |
| `EnableTemporalFlip` | Gets or sets whether temporal flip (reverse) is enabled. |
| `MaxSpeed` | Gets or sets the maximum speed factor. |
| `MinSpeed` | Gets or sets the minimum speed factor. |
| `SpatialSettings` | Gets or sets the image augmentation settings for spatial transforms. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

