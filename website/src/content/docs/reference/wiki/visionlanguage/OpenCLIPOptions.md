---
title: "OpenCLIPOptions"
description: "Configuration options for the OpenCLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the OpenCLIP model.

## For Beginners

OpenCLIP is essentially the same as CLIP but trained on publicly available
data instead of proprietary data. This makes it more transparent and reproducible while achieving
similar or better performance. It supports more model sizes and configurations.

## How It Works

OpenCLIP (Ilharco et al., 2021) is an open-source reproduction of CLIP trained on the LAION-2B
and LAION-5B datasets. It supports a wide range of vision encoder architectures (ViT-B/32 through
ViT-bigG/14) and achieves comparable or better performance than OpenAI's original CLIP.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenCLIPOptions` | Initializes a new instance with default values. |
| `OpenCLIPOptions(OpenCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dataset` | Gets or sets the pre-training dataset. |
| `LossType` | Gets or sets the contrastive loss type. |
| `Precision` | Gets or sets the precision mode for inference. |
| `UseCoCaVariant` | Gets or sets whether to use the CoCa (Contrastive Captioners) variant. |

