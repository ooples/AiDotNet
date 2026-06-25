---
title: "SegNeXtOptions"
description: "Configuration options for the SegNeXt semantic segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Semantic`

Configuration options for the SegNeXt semantic segmentation model.

## For Beginners

SegNeXt options inherit from NeuralNetworkOptions, which provides
a Seed property for reproducibility. SegNeXt uses a purely convolutional architecture
with multi-scale attention — no transformers needed — making it one of the most efficient
semantic segmentation models available.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegNeXtOptions` | Initializes a new instance with default values. |
| `SegNeXtOptions(SegNeXtOptions)` | Initializes a new instance by copying from another instance. |

