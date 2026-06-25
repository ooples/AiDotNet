---
title: "RegionCLIPOptions"
description: "Configuration options for the RegionCLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the RegionCLIP model.

## For Beginners

Regular CLIP understands whole images ("a dog in a park"), but RegionCLIP
understands specific parts of images ("the dog" vs "the park" vs "the bench"). This is useful
for tasks like object detection, where you need to understand individual objects in an image.

## How It Works

RegionCLIP (Zhong et al., CVPR 2022) extends CLIP to learn region-level (object-level) visual
representations rather than just image-level ones. It generates region-text pairs from image
captions using object proposals and learns to align individual image regions with their
corresponding text descriptions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RegionCLIPOptions` | Initializes default RegionCLIP options. |
| `RegionCLIPOptions(RegionCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Domain` | Gets or sets the domain specialization. |
| `LossType` | Gets or sets the contrastive loss type. |
| `MaxRegionsPerImage` | Gets or sets the maximum number of region proposals per image. |
| `RegionFeatureDim` | Gets or sets the region feature dimension from the RoI pooling layer. |
| `RegionTextIoUThreshold` | Gets or sets the IoU threshold for region-text assignment. |
| `RoIPoolSize` | Gets or sets the RoI (Region of Interest) pooling output size. |
| `UsePseudoLabels` | Gets or sets whether to use pseudo-labels generated from CLIP for region-text pairs. |

