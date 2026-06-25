---
title: "DARTDropoutMode"
description: "Dropout modes for DART."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Dropout modes for DART.

## Fields

| Field | Summary |
|:-----|:--------|
| `Age` | Newer trees are less likely to be dropped (protect recent trees). |
| `Uniform` | Each tree has an equal probability of being dropped. |
| `Weighted` | Trees with higher weights are more likely to be dropped. |

