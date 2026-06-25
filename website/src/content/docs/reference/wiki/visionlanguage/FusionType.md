---
title: "FusionType"
description: "Specifies how vision and language features are fused in a VLM."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies how vision and language features are fused in a VLM.

## Fields

| Field | Summary |
|:-----|:--------|
| `BridgeLayers` | Bridge layers: explicit bridge connections between encoders. |
| `CoAttention` | Co-attention: parallel streams with co-attention layers. |
| `CrossModal` | Cross-modal encoder: dedicated cross-modal transformer layers. |
| `DualStream` | Dual stream: separate encoders with cross-attention bridges. |
| `SingleStream` | Single stream: visual and text tokens concatenated in one transformer. |

