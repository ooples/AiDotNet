---
title: "SoftNMS<T>"
description: "Implements Soft-NMS algorithm which reduces confidence of overlapping boxes instead of removing them."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.PostProcessing`

Implements Soft-NMS algorithm which reduces confidence of overlapping boxes instead of removing them.

## For Beginners

Standard NMS completely removes overlapping boxes, which can be
problematic when objects are close together or occluded. Soft-NMS instead reduces the
confidence score of overlapping boxes, allowing them to potentially survive if their
confidence remains above the threshold.

## How It Works

Reference: Bodla et al., "Soft-NMS -- Improving Object Detection With One Line of Code", ICCV 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoftNMS` | Creates a new Soft-NMS instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(List<Detection<>>,Double,Double,SoftNMS<>.DecayMethod,Double)` | Applies Soft-NMS with configurable decay method. |
| `ApplyClassAware(List<Detection<>>,Double,Double,SoftNMS<>.DecayMethod,Double)` | Applies class-aware Soft-NMS. |
| `ApplyGaussian(List<Detection<>>,Double,Double)` | Applies Soft-NMS with Gaussian decay. |
| `ApplyLinear(List<Detection<>>,Double,Double)` | Applies Soft-NMS with linear decay. |

