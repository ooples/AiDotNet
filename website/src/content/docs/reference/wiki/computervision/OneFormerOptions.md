---
title: "OneFormerOptions"
description: "Configuration options for the OneFormer universal segmentation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Foundation`

Configuration options for the OneFormer universal segmentation model.

## For Beginners

OneFormer options inherit from NeuralNetworkOptions. OneFormer is trained
once on panoptic data and can perform semantic, instance, or panoptic segmentation by simply
providing a text prompt describing which task to perform. This "one model, all tasks" approach
simplifies deployment.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneFormerOptions` | Initializes a new instance with default values. |
| `OneFormerOptions(OneFormerOptions)` | Initializes a new instance by copying from another instance. |

