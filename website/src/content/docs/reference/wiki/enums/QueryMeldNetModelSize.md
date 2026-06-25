---
title: "QueryMeldNetModelSize"
description: "Defines the backbone size variants for QueryMeldNet (MQ-Former)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for QueryMeldNet (MQ-Former).

## For Beginners

QueryMeldNet uses dynamic query melding to scale across diverse
datasets. Instance queries and stuff queries are fused via cross-attention, enabling
strong generalization across multiple segmentation benchmarks.

## How It Works

**Technical Details:** Extends mask-based segmentation with a dynamic query melding
mechanism. Instance and stuff queries interact through cross-attention layers, improving
both panoptic and instance segmentation quality.

**Reference:** "QueryMeldNet: Dynamic Query Melding for Multi-Dataset Segmentation", CVPR 2025.

## Fields

| Field | Summary |
|:-----|:--------|
| `R50` | ResNet-50 backbone. |
| `SwinLarge` | Swin-L backbone. |

