---
title: "DiffCutOptions"
description: "Configuration options for the DiffCut semantic segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Semantic`

Configuration options for the DiffCut semantic segmentation model.

## For Beginners

DiffCut options inherit from NeuralNetworkOptions, which provides
a Seed property for reproducibility. DiffCut uses diffusion model features combined with
Normalized Cut graph partitioning for zero-shot semantic segmentation — no training labels needed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffCutOptions` | Initializes a new instance with default values. |
| `DiffCutOptions(DiffCutOptions)` | Initializes a new instance by copying from another instance. |

