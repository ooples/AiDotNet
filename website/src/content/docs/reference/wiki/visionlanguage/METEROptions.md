---
title: "METEROptions"
description: "Configuration options for METER (Multimodal End-to-end TransformER)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for METER (Multimodal End-to-end TransformER).

## How It Works

METER (Dou et al., CVPR 2022) is a systematic study of vision-language pre-training
components. It uses separate CLIP ViT vision encoder and RoBERTa text encoder, connected
by a co-attention transformer fusion module, providing an optimized combination of
architecture choices for VLP.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `METEROptions(METEROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumCrossAttentionLayers` | Gets or sets the number of cross-attention fusion layers. |
| `TextEncoder` | Gets or sets the text encoder type. |
| `VisionEncoder` | Gets or sets the vision encoder type. |

