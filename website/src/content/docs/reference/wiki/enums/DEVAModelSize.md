---
title: "DEVAModelSize"
description: "Defines the backbone size variants for DEVA (Decoupled Video Segmentation)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for DEVA (Decoupled Video Segmentation).

## For Beginners

DEVA separates task-specific image segmentation from temporal propagation,
enabling flexible video segmentation with any image segmentation backbone.

## How It Works

**Reference:** Cheng et al., "Tracking Anything with Decoupled Video Segmentation", ICCV 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Standard variant with ResNet-50 temporal module. |
| `Large` | Large variant with Swin-L temporal module. |

