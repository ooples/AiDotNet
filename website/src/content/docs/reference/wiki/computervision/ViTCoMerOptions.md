---
title: "ViTCoMerOptions"
description: "Configuration options for the ViT-CoMer semantic segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Semantic`

Configuration options for the ViT-CoMer semantic segmentation model.

## For Beginners

ViT-CoMer options inherit from NeuralNetworkOptions, which provides
a Seed property for reproducibility. ViT-CoMer is a hybrid model that runs CNN and transformer
branches in parallel and fuses them to get excellent boundary quality in segmentation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTCoMerOptions` | Initializes a new instance with default values. |
| `ViTCoMerOptions(ViTCoMerOptions)` | Initializes a new instance by copying from another instance. |

