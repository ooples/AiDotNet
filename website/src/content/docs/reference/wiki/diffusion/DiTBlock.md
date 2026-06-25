---
title: "DiTBlock<T>"
description: "Block structure for DiT transformer layers containing attention, MLP, and conditioning layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Block structure for DiT transformer layers containing attention, MLP, and conditioning layers.

## Methods

| Method | Summary |
|:-----|:--------|
| `EnableLowPrecisionResident` | Flags this block's large weight matrices (MLP + AdaLN + cross-attention projections) for fp16-resident inference: each is stored at half precision and upcast to fp32 transiently per forward (see `DenseLayer`), halving resident weight memory… |

