---
title: "DINOv2Options"
description: "Configuration options for the DINOv2 self-supervised vision encoder."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the DINOv2 self-supervised vision encoder.

## How It Works

DINOv2 (Oquab et al., 2024) trains ViT with self-supervised objectives (iBOT + DINO head)
on 142M curated images (LVD-142M). It produces universal visual features without labels,
achieving linear-probe results competitive with fine-tuned CLIP on many benchmarks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DINOv2Options(DINOv2Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DINOHeadDim` | Gets or sets the self-supervised head dimension for DINO loss. |
| `IBOTMaskRatio` | Gets or sets the iBOT mask ratio for masked image modeling. |
| `NumRegisterTokens` | Gets or sets the number of register tokens appended to the patch sequence. |
| `UseRegisterTokens` | Gets or sets whether to use register tokens (DINOv2 with registers variant). |

