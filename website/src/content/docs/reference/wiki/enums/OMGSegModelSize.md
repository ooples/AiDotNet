---
title: "OMGSegModelSize"
description: "Defines the backbone size variants for OMG-Seg."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for OMG-Seg.

## For Beginners

OMG-Seg (One Model that is Great for all Segmentation) handles over
10 different segmentation tasks with a single model and only 70M trainable parameters.
It uses task-specific queries to switch between tasks at inference time.

## How It Works

**Technical Details:** Uses a shared transformer backbone with task-specific query sets.
Supports image segmentation (semantic, instance, panoptic), video segmentation, open-vocabulary
segmentation, interactive segmentation, and more.

**Reference:** Li et al., "OMG-Seg: Is One Model Good Enough For All Segmentation?", CVPR 2024.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base variant (~70M trainable params). |
| `Large` | Large variant (~120M trainable params). |

