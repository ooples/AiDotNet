---
title: "SAMHQModelSize"
description: "Defines the backbone size variants for SAM-HQ (High-Quality Segment Anything Model)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for SAM-HQ (High-Quality Segment Anything Model).

## For Beginners

SAM-HQ extends the Segment Anything Model (SAM) with a High-Quality
output token that produces significantly sharper and more accurate mask boundaries. It uses
the same ViT backbone sizes as the original SAM.

## How It Works

**Technical Details:** SAM-HQ adds an HQ output token and learnable global-local feature
fusion to the original SAM architecture. The backbone is a Vision Transformer (ViT) with
varying sizes. Training uses only 44K fine-grained masks.

**Reference:** Ke et al., "Segment Anything in High Quality", NeurIPS 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `ViTBase` | ViT-B backbone (91M params). |
| `ViTHuge` | ViT-H backbone (636M params). |
| `ViTLarge` | ViT-L backbone (308M params). |

