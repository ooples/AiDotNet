---
title: "YOLOv9<T>"
description: "YOLOv9 object detector with Programmable Gradient Information (PGI)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO`

YOLOv9 object detector with Programmable Gradient Information (PGI).

## For Beginners

YOLOv9 introduces Programmable Gradient Information (PGI)
and Generalized Efficient Layer Aggregation Network (GELAN) to address information
loss during feature transformation, achieving state-of-the-art performance.

## How It Works

Key features:

- PGI: Programmable Gradient Information for better gradient flow
- GELAN: Generalized ELAN for efficient feature aggregation
- Auxiliary reversible branch for improved training
- Better information preservation through the network

Reference: Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv9(NeuralNetworkArchitecture<>,ModelSize,Int32)` | Creates a new YOLOv9 detector with default options derived from the architecture. |
| `YOLOv9(ObjectDetectionOptions<>)` | Creates a new YOLOv9 detector. |

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

