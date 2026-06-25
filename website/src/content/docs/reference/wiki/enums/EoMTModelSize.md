---
title: "EoMTModelSize"
description: "Defines the backbone size variants for EoMT (Encoder-only Mask Transformer)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for EoMT (Encoder-only Mask Transformer).

## For Beginners

EoMT removes the complex pixel decoder and transformer decoder used
by models like Mask2Former, instead placing mask queries directly inside a plain Vision
Transformer (ViT/DINOv2). This makes it 4.4x faster than Mask2Former while maintaining
competitive accuracy.

## How It Works

**Technical Details:** Uses DINOv2 as the backbone. Queries are inserted at intermediate
ViT layers and processed alongside image tokens. No separate decoder needed. Achieves strong
results on COCO panoptic, ADE20K semantic, and Cityscapes instance.

**Reference:** Saporta et al., "Encoder-only Mask Transformer", CVPR 2025 Highlight.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | ViT-B/DINOv2 backbone (~86M params). |
| `Large` | ViT-L/DINOv2 backbone (~307M params). |
| `Small` | ViT-S/DINOv2 backbone (~22M params). |

