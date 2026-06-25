---
title: "GroundingVLMOptions"
description: "Base configuration options for visual grounding models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Grounding`

Base configuration options for visual grounding models.

## For Beginners

These options configure the Grounding model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroundingVLMOptions` | Initializes a new instance with default values. |
| `GroundingVLMOptions(GroundingVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BoxDimension` | Gets or sets the number of output coordinates per box (typically 4 for [x1,y1,x2,y2]). |
| `ConfidenceThreshold` | Gets or sets the confidence threshold for detection filtering. |
| `MaxDetections` | Gets or sets the maximum number of detections per image. |
| `NmsThreshold` | Gets or sets the IoU threshold for non-maximum suppression. |

