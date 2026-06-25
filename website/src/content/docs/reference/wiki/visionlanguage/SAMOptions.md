---
title: "SAMOptions"
description: "Configuration options for the Segment Anything Model (SAM) vision encoder."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the Segment Anything Model (SAM) vision encoder.

## How It Works

SAM (Kirillov et al., 2023) consists of a ViT-based image encoder that produces image embeddings,
which can be combined with prompt embeddings (points, boxes, masks) via a lightweight mask decoder
for promptable segmentation of any object.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAMOptions(SAMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaskDecoderDim` | Gets or sets the mask decoder embedding dimension. |
| `MaxPointsPerPrompt` | Gets or sets the maximum number of points per prompt. |
| `NumMaskDecoderLayers` | Gets or sets the number of mask decoder layers. |
| `NumMultimaskOutputs` | Gets or sets the number of multimask outputs. |
| `UseRelativePositionalEncoding` | Gets or sets whether to use relative positional encoding (window attention). |
| `WindowSize` | Gets or sets the window size for windowed attention in the encoder. |

