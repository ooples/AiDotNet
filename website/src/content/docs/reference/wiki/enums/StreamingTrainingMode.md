---
title: "StreamingTrainingMode"
description: "Controls the memory-bounded streaming training path (optimizer-in-backward with 8-bit Adam state and topological-min gradient release)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Controls the memory-bounded streaming training path (optimizer-in-backward
with 8-bit Adam state and topological-min gradient release).

## For Beginners

Very large models can run out of memory during
training because the gradients and optimizer state are several times the size
of the model itself. Streaming training applies each parameter's update the
instant its gradient is ready and then frees that gradient, so the full
gradient set never has to fit in memory at once. This setting decides when
that path is used.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Engage streaming automatically: the autotuner turns it on only when the model's estimated full-precision training footprint would not comfortably fit in available memory. |
| `ForceOff` | Never use the streaming training path (classic in-memory training only). |
| `ForceOn` | Always use the streaming training path (mainly for tests). |

