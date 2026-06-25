---
title: "FusedOptimizerPathEvent"
description: "Emitted when `TrainWithTape` enters or skips the fused-compiled fast path."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Emitted when `TrainWithTape` enters or skips the fused-compiled
fast path. Hit means forward + backward + optimizer step ran in a
single compiled kernel; miss falls back to the eager tape walk.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FusedOptimizerPathEvent(Int32,Boolean,String)` | Emitted when `TrainWithTape` enters or skips the fused-compiled fast path. |

