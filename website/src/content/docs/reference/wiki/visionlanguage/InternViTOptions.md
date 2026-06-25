---
title: "InternViTOptions"
description: "Configuration options for InternViT, the vision encoder used in the InternVL series."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for InternViT, the vision encoder used in the InternVL series.

## How It Works

InternViT (Chen et al., 2024) is a 6B-parameter ViT designed for progressive alignment
with LLMs. It uses dynamic resolution processing and pixel shuffle downsampling to handle
images of varying sizes efficiently as part of the InternVL architecture.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InternViTOptions(InternViTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTiles` | Gets or sets the maximum number of tiles for dynamic resolution. |
| `PixelShuffleRatio` | Gets or sets the pixel shuffle downsampling ratio. |
| `Use3DRoPE` | Gets or sets whether to use 3D-RoPE for positional encoding. |
| `UseDynamicResolution` | Gets or sets whether to use dynamic resolution tiling. |

