---
title: "DETR<T>"
description: "DETR (DEtection TRansformer) - End-to-end object detection with transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.DETR`

DETR (DEtection TRansformer) - End-to-end object detection with transformers.

## For Beginners

DETR is a revolutionary approach to object detection that uses
a transformer architecture instead of traditional anchor-based methods. It treats detection
as a set prediction problem, eliminating the need for NMS post-processing.

## How It Works

Key features:

- No anchors needed
- No NMS required (uses Hungarian matching for training)
- Global reasoning via self-attention
- Simple end-to-end architecture

Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DETR(ObjectDetectionOptions<>)` | Creates a new DETR detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>,Double,Double)` |  |
| `Forward(Tensor<>)` |  |
| `GetHeadParameterCount` |  |
| `GetSizeConfig(ModelSize)` | Gets configuration based on model size. |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double,Double)` |  |
| `SaveWeights(String)` |  |

