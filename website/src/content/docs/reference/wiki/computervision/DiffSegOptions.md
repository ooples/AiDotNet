---
title: "DiffSegOptions"
description: "Configuration options for the DiffSeg semantic segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Semantic`

Configuration options for the DiffSeg semantic segmentation model.

## For Beginners

DiffSeg options inherit from NeuralNetworkOptions, which provides
a Seed property for reproducibility. DiffSeg produces unsupervised segmentation by merging
self-attention maps from a diffusion model, requiring no training labels at all.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffSegOptions` | Initializes a new instance with default values. |
| `DiffSegOptions(DiffSegOptions)` | Initializes a new instance by copying from another instance. |

