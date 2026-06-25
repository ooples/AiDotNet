---
title: "SigLIPSOOptions"
description: "Configuration options for SigLIP-SO (Shape-Optimized SigLIP), a 400M vision encoder."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for SigLIP-SO (Shape-Optimized SigLIP), a 400M vision encoder.

## How It Works

SigLIP-SO (Zhai et al., 2023) is a shape-optimized variant of SigLIP designed for
use as a standalone vision encoder in VLMs. The SO-400M version (ViT-SO400M/14) uses
an optimized width/depth ratio for the 400M parameter budget, producing high-quality
visual features widely adopted in LLaVA, PaliGemma, and other VLMs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SigLIPSOOptions(SigLIPSOOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumOutputTokens` | Gets or sets the number of output feature tokens after pooling. |
| `TrainingResolution` | Gets or sets the resolution at which the model was trained. |
| `UseSigmoidLoss` | Gets or sets whether to use sigmoid loss (SigLIP) for any fine-tuning. |

