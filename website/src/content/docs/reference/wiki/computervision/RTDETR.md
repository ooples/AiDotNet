---
title: "RTDETR<T>"
description: "RT-DETR (Real-Time DEtection TRansformer) - First real-time end-to-end object detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.DETR`

RT-DETR (Real-Time DEtection TRansformer) - First real-time end-to-end object detector.

## For Beginners

RT-DETR is the first real-time end-to-end transformer-based
object detector. It achieves YOLO-level speed while maintaining transformer accuracy
by using a hybrid encoder and efficient decoder design.

## How It Works

Key features:

- Hybrid encoder with intra-scale and cross-scale feature interaction
- Efficient decoder with uncertainty-minimal query selection
- Flexible inference speed/accuracy trade-off
- No NMS post-processing required

Reference: Lv et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RTDETR(ObjectDetectionOptions<>)` | Creates a new RT-DETR detector. |

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
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double,Double)` |  |
| `SaveWeights(String)` |  |

