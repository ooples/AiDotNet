---
title: "FastSAMOptions"
description: "Configuration options for FastSAM (CNN-based fast Segment Anything)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Efficient`

Configuration options for FastSAM (CNN-based fast Segment Anything).

## For Beginners

These options configure the FastSAM model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastSAMOptions` | Initializes a new instance with default values. |
| `FastSAMOptions(FastSAMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChannelDims` | Channel dimensions for each backbone stage. |
| `DecoderDim` | Decoder hidden dimension. |
| `Depths` | Number of blocks per backbone stage. |

