---
title: "ViTAdapterOptions"
description: "Configuration options for the ViT-Adapter semantic segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Semantic`

Configuration options for the ViT-Adapter semantic segmentation model.

## For Beginners

ViT-Adapter options inherit from NeuralNetworkOptions, which provides
a Seed property for reproducibility. ViT-Adapter enables plain Vision Transformers to handle
dense prediction tasks by adding lightweight spatial prior modules, without requiring any
vision-specific architectural changes to the base ViT.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTAdapterOptions` | Initializes a new instance with default values. |
| `ViTAdapterOptions(ViTAdapterOptions)` | Initializes a new instance by copying from another instance. |

