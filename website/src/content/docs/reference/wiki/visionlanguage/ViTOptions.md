---
title: "ViTOptions"
description: "Configuration options for the Vision Transformer (ViT) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the Vision Transformer (ViT) model.

## How It Works

ViT (Dosovitskiy et al., ICLR 2021) splits an image into fixed-size patches, linearly
embeds them, adds position embeddings, and processes the sequence through a standard Transformer
encoder. A [CLS] token aggregates information for classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTOptions(ViTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PositionalEmbedding` | Gets or sets the positional embedding type. |
| `UseClsToken` | Gets or sets whether to use a [CLS] token for classification. |
| `Variant` | Gets or sets the ViT model variant. |

