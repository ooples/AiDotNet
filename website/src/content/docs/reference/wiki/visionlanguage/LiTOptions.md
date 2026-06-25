---
title: "LiTOptions"
description: "Configuration options for the LiT (Locked-image Tuning) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the LiT (Locked-image Tuning) model.

## For Beginners

LiT takes a shortcut: instead of training both image and text encoders from
scratch (which is expensive), it takes an already-trained image model and only teaches the text model
to align with it. This is much faster and often works just as well.

## How It Works

LiT (Zhai et al., CVPR 2022) freezes a pre-trained image encoder and only trains the text encoder
and projection layers. This "locked-image tuning" approach achieves strong zero-shot performance
while being much cheaper to train than full contrastive learning from scratch.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiTOptions` | Initializes default LiT options. |
| `LiTOptions(LiTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FreezeVisionEncoder` | Gets or sets whether to freeze the vision encoder during training. |
| `InitializeTextFromPretrained` | Gets or sets whether to use a pre-trained text encoder as initialization. |
| `LossType` | Gets or sets the contrastive loss type. |
| `PretrainedVisionWeightsPath` | Gets or sets the path to the pre-trained vision encoder weights. |

