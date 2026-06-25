---
title: "InternImageOptions"
description: "Configuration options for the InternImage semantic segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Semantic`

Configuration options for the InternImage semantic segmentation model.

## For Beginners

InternImage options inherit from NeuralNetworkOptions, which provides
a Seed property for reproducibility. InternImage is a large-scale CNN that uses Deformable
Convolution v3 (DCNv3) to compete with Vision Transformers on dense prediction tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InternImageOptions` | Initializes a new instance with default values. |
| `InternImageOptions(InternImageOptions)` | Initializes a new instance by copying from another instance. |

