---
title: "SOLOv2<T>"
description: "SOLOv2 (Segmenting Objects by Locations v2) for instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

SOLOv2 (Segmenting Objects by Locations v2) for instance segmentation.

## For Beginners

SOLOv2 is an anchor-free, box-free instance segmentation
method. It directly predicts instance masks by location, dividing the image into a grid
and predicting masks for each cell. This eliminates the need for ROI operations.

## How It Works

Key features:

- Direct mask prediction without boxes
- Dynamic convolution for mask generation
- Grid-based location encoding
- More efficient than two-stage methods

Reference: Wang et al., "SOLOv2: Dynamic and Fast Instance Segmentation", NeurIPS 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SOLOv2(InstanceSegmentationOptions<>)` | Creates a new SOLOv2 model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `SaveWeights(String)` |  |
| `Segment(Tensor<>)` |  |

