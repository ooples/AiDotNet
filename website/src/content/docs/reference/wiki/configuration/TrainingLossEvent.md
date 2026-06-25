---
title: "TrainingLossEvent"
description: "Emitted once per call to `TrainWithTape` with the scalar loss value returned by the loss function for that batch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Emitted once per call to `TrainWithTape` with the scalar loss
value returned by the loss function for that batch. Use to track
loss trajectory across epochs without hand-instrumenting the loop.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainingLossEvent(Int32,Double,Int32,Int64)` | Emitted once per call to `TrainWithTape` with the scalar loss value returned by the loss function for that batch. |

