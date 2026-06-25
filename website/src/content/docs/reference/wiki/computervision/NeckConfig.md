---
title: "NeckConfig"
description: "Configuration for neck modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Detection.Necks`

Configuration for neck modules.

## Properties

| Property | Summary |
|:-----|:--------|
| `Activation` | Activation function to use (e.g., "relu", "silu", "gelu"). |
| `InputChannels` | Input channels from the backbone at each level. |
| `NumLevels` | Number of feature pyramid levels. |
| `OutputChannels` | Number of output channels for all feature levels. |
| `UseBatchNorm` | Whether to use batch normalization. |
| `UseExtraConvs` | Whether to add extra convolution layers for feature refinement. |

