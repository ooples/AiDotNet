---
title: "NnUNetModelSize"
description: "Defines the architecture variants for nnU-Net v2 medical segmentation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the architecture variants for nnU-Net v2 medical segmentation.

## For Beginners

nnU-Net automatically configures everything (architecture, preprocessing,
training, post-processing) per dataset. These variants control the base architecture.

## How It Works

**Reference:** Isensee et al., "nnU-Net: a self-configuring method for deep learning-based
biomedical image segmentation", Nature Methods 2021.

## Fields

| Field | Summary |
|:-----|:--------|
| `UNet2D` | 2D U-Net. |
| `UNet3DCascade` | 3D low-resolution U-Net + cascade. |
| `UNet3DFull` | 3D full-resolution U-Net. |

