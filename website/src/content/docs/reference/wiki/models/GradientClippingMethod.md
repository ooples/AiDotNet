---
title: "GradientClippingMethod"
description: "Specifies the method used for gradient clipping."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the method used for gradient clipping.

## Fields

| Field | Summary |
|:-----|:--------|
| `ByNorm` | Clips gradients by scaling the entire gradient vector if its L2 norm exceeds a threshold. |
| `ByValue` | Clips each gradient element independently to a fixed range. |

