---
title: "EoMTOptions"
description: "Configuration options for the EoMT (Encoder-only Mask Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Foundation`

Configuration options for the EoMT (Encoder-only Mask Transformer) model.

## For Beginners

EoMT removes the pixel and transformer decoders used by Mask2Former,
placing mask queries directly inside a plain ViT (DINOv2). This yields 4.4x faster inference.
Options inherit from NeuralNetworkOptions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EoMTOptions` | Initializes a new instance with default values. |
| `EoMTOptions(EoMTOptions)` | Initializes a new instance by copying from another instance. |

