---
title: "RPN<T>"
description: "Region Proposal Network (RPN) - Generates object proposals for two-stage detectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN`

Region Proposal Network (RPN) - Generates object proposals for two-stage detectors.

## For Beginners

RPN is a neural network that scans an image and proposes
regions that likely contain objects. It's the first stage in two-stage detectors like
Faster R-CNN, enabling end-to-end training.

## How It Works

Key features:

- Slides a small network over the feature map
- Generates proposals at multiple scales and aspect ratios via anchors
- Predicts objectness scores and bounding box refinements
- Shared features with detection network for efficiency

Reference: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
Region Proposal Networks", NeurIPS 2015

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RPN(Int32,Int32,Int32[],Double[],Nullable<Int32>)` | Creates a new Region Proposal Network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AnchorGenerator` | Gets the anchor generator used by this RPN. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Forward pass through the RPN. |
| `GenerateProposals(Tensor<>,Tensor<>,List<BoundingBox<>>,Int32,Int32,Int32,Int32,Double)` | Generates proposals from RPN outputs. |
| `GetParameterCount` | Gets the total parameter count for this RPN. |
| `ReadParameters(BinaryReader)` | Reads the RPN parameters from a binary stream. |
| `WriteParameters(BinaryWriter)` | Writes the RPN parameters to a binary stream. |

