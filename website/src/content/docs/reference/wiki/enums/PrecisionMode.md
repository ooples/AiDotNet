---
title: "PrecisionMode"
description: "Defines the numeric precision mode for neural network training and computation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the numeric precision mode for neural network training and computation.

## Fields

| Field | Summary |
|:-----|:--------|
| `BF16` | Brain float 16 (bfloat16) format. |
| `FP16` | Half precision using 16-bit floating-point (Half/FP16). |
| `FP32` | Full precision using 32-bit floating-point (float/FP32). |
| `FP64` | Double precision using 64-bit floating-point (double/FP64). |
| `Mixed` | Mixed precision training: FP16 for forward/backward passes, FP32 for parameter updates. |

