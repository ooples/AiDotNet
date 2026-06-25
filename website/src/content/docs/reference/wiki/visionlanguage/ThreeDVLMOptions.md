---
title: "ThreeDVLMOptions"
description: "Base configuration options for 3D vision-language models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.ThreeD`

Base configuration options for 3D vision-language models.

## For Beginners

These options configure the ThreeD model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThreeDVLMOptions` | Initializes a new instance with default values. |
| `ThreeDVLMOptions(ThreeDVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets or sets the language model backbone name. |
| `MaxPoints` | Gets or sets the maximum number of 3D points the model can process. |
| `PointChannels` | Gets or sets the number of channels per point (3=XYZ, 6=XYZ+RGB, 9=XYZ+RGB+normals). |
| `PointEncoderDim` | Gets or sets the point cloud encoder hidden dimension. |

