---
title: "DIoUNMS<T>"
description: "Implements Distance-IoU based Non-Maximum Suppression for improved localization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.PostProcessing`

Implements Distance-IoU based Non-Maximum Suppression for improved localization.

## For Beginners

DIoU-NMS is an improved version of standard NMS that considers
the distance between box centers, not just their overlap. This helps preserve nearby
objects that might be suppressed by standard NMS.

## How It Works

Reference: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DIoUNMS` | Creates a new DIoU-NMS instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(List<Detection<>>,Double)` | Applies DIoU-NMS to a list of detections. |
| `ApplyAdaptive(List<Detection<>>,Double,Double)` | Applies DIoU-NMS with adaptive threshold based on box density. |
| `ApplyBatched(List<List<Detection<>>>,Double,Boolean)` | Applies batched DIoU-NMS for multiple images. |
| `ApplyClassAware(List<Detection<>>,Double)` | Applies class-aware DIoU-NMS. |

