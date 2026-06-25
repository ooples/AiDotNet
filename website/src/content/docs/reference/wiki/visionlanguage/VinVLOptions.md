---
title: "VinVLOptions"
description: "Configuration options for VinVL (Visual Features in Vision-Language)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for VinVL (Visual Features in Vision-Language).

## How It Works

VinVL (Zhang et al., CVPR 2021) improves on Oscar by providing better visual features
through a stronger object detection backbone (ResNeXt-152 C4), enriching object tags with
attributes and achieving state-of-the-art on VQA, captioning, and retrieval benchmarks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VinVLOptions(VinVLOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectionThreshold` | Gets or sets the object detection confidence threshold. |
| `MaxImageRegions` | Gets or sets the maximum number of image regions. |
| `MaxObjectTags` | Gets or sets the maximum number of object tags with attributes. |
| `UseAttributePredictions` | Gets or sets whether to include attribute predictions in object tags. |

