---
title: "FasterRCNN<T>"
description: "Faster R-CNN - Two-stage object detection with region proposal network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN`

Faster R-CNN - Two-stage object detection with region proposal network.

## For Beginners

Faster R-CNN is a foundational object detection model that
works in two stages: first proposing regions that might contain objects (RPN),
then classifying and refining those proposals. It's highly accurate but slower
than single-stage detectors like YOLO.

## How It Works

Key features:

- Two-stage detection: RPN + Fast R-CNN
- End-to-end trainable
- High accuracy through refined proposals
- RoI pooling/align for fixed-size feature extraction

Reference: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
Region Proposal Networks", NeurIPS 2015

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FasterRCNN(NeuralNetworkArchitecture<>,ModelSize,Int32)` | Creates a new Faster R-CNN detector with default options derived from the architecture. |
| `FasterRCNN(ObjectDetectionOptions<>)` | Creates a new Faster R-CNN detector. |

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

