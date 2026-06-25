---
title: "BASICOptions"
description: "Configuration options for the BASIC (Batch-wise Alignment of Scaled Image-text Contrastive) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the BASIC (Batch-wise Alignment of Scaled Image-text Contrastive) model.

## For Beginners

BASIC is a scaled-up version of CLIP that uses a hybrid CNN-Transformer for images
(combining the strengths of both architectures) and was trained on an even larger dataset. The name
"BASIC" highlights that even a basic contrastive approach works extremely well at sufficient scale.

## How It Works

BASIC (Pham et al., 2022) scales up the dual-encoder contrastive learning paradigm using a CoAtNet
(hybrid CNN-Transformer) as the vision encoder and a large text transformer, training on 6.6 billion
image-text pairs. It achieves 85.7% zero-shot accuracy on ImageNet.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BASICOptions` | Initializes default BASIC options. |
| `BASICOptions(BASICOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CoAtNetVariant` | Gets or sets the CoAtNet variant to use as the vision encoder. |
| `LossType` | Gets or sets the contrastive loss type. |
| `UseGradientCheckpointing` | Gets or sets whether to use gradient checkpointing for memory-efficient training. |

