---
title: "ISegmentationModel<T>"
description: "Base interface for all image segmentation models that classify pixels into categories."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Base interface for all image segmentation models that classify pixels into categories.

## For Beginners

Segmentation is like coloring a picture where each color represents
a different object or category. Unlike object detection which draws boxes around things,
segmentation gives you the exact pixel-level outline.

Types of segmentation:

- Semantic: "Which pixels are road? Which are sky?" (classes, no instances)
- Instance: "Where is car #1? Car #2?" (individual objects)
- Panoptic: Both semantic + instance together
- Interactive: You point/click and the model segments what you indicated

Common use cases:

- Autonomous driving (road, lane, obstacle detection)
- Medical imaging (organ and tumor boundaries)
- Photo editing (background removal, object selection)
- Agriculture (crop vs. weed detection from drones)

## How It Works

Image segmentation assigns a label to every pixel in an image, enabling fine-grained
scene understanding. This is the foundation interface that all segmentation model types
(semantic, instance, panoptic, medical, etc.) extend.

## Properties

| Property | Summary |
|:-----|:--------|
| `InputHeight` | Gets the expected input image height in pixels. |
| `InputWidth` | Gets the expected input image width in pixels. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `NumClasses` | Gets the number of segmentation classes this model predicts. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Segment(Tensor<>)` | Segments an image, producing a per-pixel class prediction map. |

