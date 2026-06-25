---
title: "ViTVariant"
description: "Specifies the Vision Transformer (ViT) variant used as the image encoder."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies the Vision Transformer (ViT) variant used as the image encoder.

## Fields

| Field | Summary |
|:-----|:--------|
| `CoAtNet` | CoAtNet: Hybrid CNN-Transformer used by BASIC. |
| `EfficientNetB7` | EfficientNet-B7: CNN backbone used by ALIGN. |
| `ViTB16` | ViT-B/16: Base model with 16x16 patch size (86M params, better than B/32). |
| `ViTB32` | ViT-B/32: Base model with 32x32 patch size (86M params, fastest). |
| `ViTBigG14` | ViT-bigG/14: Biggest model with 14x14 patch size (2.54B params). |
| `ViTE14` | ViT-e/14: EVA-CLIP extra-large model (4.4B params). |
| `ViTG14` | ViT-G/14: Giant model with 14x14 patch size (1.01B params). |
| `ViTH14` | ViT-H/14: Huge model with 14x14 patch size (632M params). |
| `ViTL14` | ViT-L/14: Large model with 14x14 patch size (304M params). |
| `ViTL14At336` | ViT-L/14@336: Large model at 336px resolution. |
| `ViTSO400M14` | ViT-SO400M/14: SigLIP Shape-Optimized 400M param model. |

