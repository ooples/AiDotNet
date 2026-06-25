---
title: "BlendingMode"
description: "Blending modes for overlapping window regions."
section: "API Reference"
---

`Enums` · `AiDotNet.Diffusion.Acceleration`

Blending modes for overlapping window regions.

## Fields

| Field | Summary |
|:-----|:--------|
| `CosineBlend` | Cosine-weighted blending for smoother transitions. |
| `HardCut` | Use the latest window's frames (hard cut). |
| `LinearBlend` | Linear interpolation between overlapping frames. |
| `SlerpBlend` | Smooth ease-in/ease-out (S-curve) blending in latent space. |

