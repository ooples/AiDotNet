---
title: "SAMModelSize"
description: "Defines the backbone size variants for SAM (Segment Anything Model)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for SAM (Segment Anything Model).

## For Beginners

SAM is the original Segment Anything Model from Meta AI. It uses a
Vision Transformer (ViT) backbone with varying sizes. Larger backbones produce more accurate
masks but are slower. Choose ViTBase for a good balance, or ViTHuge for maximum accuracy.

## How It Works

**Technical Details:** The ViT encoder processes images at 1024x1024 resolution using
16x16 patches. The model was trained on the SA-1B dataset containing 1B+ masks.

**Reference:** Kirillov et al., "Segment Anything", ICCV 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `ViTBase` | ViT-B backbone (91M params). |
| `ViTHuge` | ViT-H backbone (636M params). |
| `ViTLarge` | ViT-L backbone (308M params). |

