---
title: "MaskVisualizer<T>"
description: "Visualizes instance segmentation results on images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Visualization`

Visualizes instance segmentation results on images.

## For Beginners

This class overlays colored masks and bounding boxes
on images to visualize instance segmentation results.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskVisualizer(VisualizationOptions)` | Creates a new mask visualizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstanceIdMap(InstanceSegmentationResult<>,Int32,Int32)` | Creates an instance ID map where each pixel contains the instance index. |
| `CreateSemanticMap(InstanceSegmentationResult<>,Int32,Int32)` | Creates a combined semantic segmentation map from instances. |
| `Visualize(Tensor<>,InstanceSegmentationResult<>,String[])` | Draws instance segmentation results on an image. |
| `VisualizeMasksOnly(Tensor<>,InstanceSegmentationResult<>)` | Draws only the mask overlay without bounding boxes. |

