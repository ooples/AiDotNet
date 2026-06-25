---
title: "DINOv3Options"
description: "Configuration options for DINOv3, the next evolution of self-supervised vision encoders."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for DINOv3, the next evolution of self-supervised vision encoders.

## How It Works

DINOv3 (Meta, 2025) scales self-supervised ViT to 7B parameters trained on 1.7B images,
outperforming SigLIP 2 on most vision benchmarks. It introduces improved training recipes
with enhanced data augmentation and longer training schedules.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DINOv3Options(DINOv3Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DINOHeadDim` | Gets or sets the self-supervised head dimension. |
| `IBOTMaskRatio` | Gets or sets the iBOT mask ratio. |
| `NumRegisterTokens` | Gets or sets the number of register tokens. |
| `UseRegisterTokens` | Gets or sets whether to use register tokens. |
| `UseSwiGLU` | Gets or sets whether to use SwiGLU activation in the FFN. |

