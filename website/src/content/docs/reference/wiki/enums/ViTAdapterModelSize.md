---
title: "ViTAdapterModelSize"
description: "Defines the model size variants for ViT-Adapter."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the model size variants for ViT-Adapter.

## For Beginners

ViT-Adapter adds spatial prior modules to plain Vision Transformers
so they can handle dense prediction tasks like segmentation. The sizes correspond to
different ViT backbone sizes (Small, Base, Large).

## How It Works

**Technical Details:** Each size uses a different ViT backbone with adapter modules
that inject multi-scale spatial priors. The adapter modules are lightweight and add only
a small percentage of extra parameters on top of the base ViT.

**Reference:** Chen et al., "Vision Transformer Adapter for Dense Predictions",
ICLR 2023 Spotlight.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | ViT-Adapter-B: Base variant based on ViT-Base (86M params). |
| `Large` | ViT-Adapter-L: Large variant based on ViT-Large (304M params). |
| `Small` | ViT-Adapter-S: Small variant based on ViT-Small (48M params). |

