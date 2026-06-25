---
title: "EAST<T>"
description: "EAST (Efficient and Accurate Scene Text) detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.TextDetection`

EAST (Efficient and Accurate Scene Text) detector.

## For Beginners

EAST is a fast and accurate text detector that directly
predicts word or text-line bounding boxes in one forward pass. It outputs a score map
showing where text is likely to be, plus geometry (box coordinates or rotated rectangles)
for each text region.

## How It Works

Key features:

- Single-shot detection (no region proposals)
- Supports both axis-aligned and rotated bounding boxes
- Fast inference suitable for real-time applications
- Works well for both horizontal and multi-oriented text

Reference: Zhou et al., "EAST: An Efficient and Accurate Scene Text Detector", CVPR 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EAST(TextDetectionOptions<>,Boolean)` | Creates a new EAST text detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,Double)` |  |
| `Forward(Tensor<>)` |  |
| `GetHeadParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double)` |  |
| `SaveWeights(String)` |  |

