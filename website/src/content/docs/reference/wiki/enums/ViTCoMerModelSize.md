---
title: "ViTCoMerModelSize"
description: "Defines the model size variants for ViT-CoMer (hybrid CNN-Transformer)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the model size variants for ViT-CoMer (hybrid CNN-Transformer).

## For Beginners

ViT-CoMer combines a CNN branch with a Vision Transformer branch
to get the best of both worlds — CNNs excel at fine local details while transformers
capture global context. The result is better boundary quality in segmentation.

## How It Works

**Technical Details:** Each size uses a different ViT backbone paired with a CNN branch
of matching capacity. The two branches exchange information through multi-scale feature
interaction modules.

**Reference:** Xia et al., "ViT-CoMer: Vision Transformer with Convolutional
Multi-scale Feature Interaction for Dense Predictions", CVPR 2024.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | ViT-CoMer-B: Base variant (100M params). |
| `Large` | ViT-CoMer-L: Large variant (350M params). |
| `Small` | ViT-CoMer-S: Small variant (45M params). |

