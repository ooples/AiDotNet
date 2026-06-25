---
title: "SymbolicShapeMode"
description: "Strategy for how `CompiledModelHost` keys the compile cache."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks`

Strategy for how `CompiledModelHost` keys the compile cache. Dynamic
dims let one compiled plan serve multiple concrete shapes — essential for bursty
inference traffic where every request has a different batch size.

## Fields

| Field | Summary |
|:-----|:--------|
| `AllDynamic` | Every dim dynamic. |
| `BatchAndSeqDynamic` | Dims 0 (batch) and 1 (seq-len) dynamic. |
| `BatchDynamic` | Dim 0 (batch) dynamic; all others static. |
| `Static` | Every dim treated as static. |

