---
title: "ReplayBufferStrategy"
description: "Specifies the buffer management strategy for Experience Replay."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the buffer management strategy for Experience Replay.

## Fields

| Field | Summary |
|:-----|:--------|
| `ClassBalanced` | Class-balanced sampling - maintains equal samples per class. |
| `Reservoir` | Reservoir sampling - uniform random replacement ensuring equal probability. |
| `Ring` | Ring buffer - FIFO queue where oldest samples are removed first. |

