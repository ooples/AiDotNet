---
title: "XDecoderModelSize"
description: "Defines the backbone size variants for X-Decoder."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for X-Decoder.

## For Beginners

X-Decoder is a generalist vision decoder that handles referring segmentation,
open-vocabulary segmentation, and image captioning in a single unified model. It decodes both
pixel-level masks and text tokens using one shared architecture.

## How It Works

**Technical Details:** Uses a two-path decoder: one for pixel-level predictions (masks) and
one for token-level predictions (text). Both paths share the same attention mechanism. Supports
various segmentation and vision-language tasks without task-specific heads.

**Reference:** Zou et al., "Generalized Decoding for Pixel, Image, and Language", CVPR 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base backbone (~86M params). |
| `Large` | Large backbone (~307M params). |
| `Tiny` | Tiny backbone (~30M params). |

