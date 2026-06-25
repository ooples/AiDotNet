---
title: "ViLTOptions"
description: "Configuration options for ViLT (Vision-and-Language Transformer)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for ViLT (Vision-and-Language Transformer).

## How It Works

ViLT (Kim et al., ICML 2021) is a minimal architecture that removes the CNN/object detector
entirely. Raw image patches are linearly embedded and concatenated with text tokens in a single
transformer, making it 60x faster than region-feature-based models at comparable accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViLTOptions(ViLTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PatchSize` | Gets or sets the patch size for image tokenization. |
| `UseRandAugment` | Gets or sets whether to use image augmentation during training. |
| `UseWholeWordMasking` | Gets or sets whether to use whole-word masking for MLM pre-training. |

