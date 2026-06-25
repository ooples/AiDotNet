---
title: "LayerApiShape"
description: "Describes the Forward method signature shape a layer uses."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Describes the Forward method signature shape a layer uses.
The test scaffold generator uses this to select the correct test base class.

## Fields

| Field | Summary |
|:-----|:--------|
| `DualTensor` | Dual-input Forward(Tensor, Tensor) → Tensor interface. |
| `GraphWithSetup` | Standard Forward(Tensor) but requires graph setup before use (adjacency matrix, Laplacian, eigenbasis, etc.). |
| `MultiInput` | Multi-input Forward(params Tensor[]) → Tensor interface. |
| `SingleTensor` | Standard Forward(Tensor) → Tensor interface. |

