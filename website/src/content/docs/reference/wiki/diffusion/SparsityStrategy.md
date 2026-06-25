---
title: "SparsityStrategy"
description: "Strategy for selecting which transformer blocks to skip in sparse computation."
section: "API Reference"
---

`Enums` · `AiDotNet.Diffusion.Acceleration`

Strategy for selecting which transformer blocks to skip in sparse computation.

## Fields

| Field | Summary |
|:-----|:--------|
| `AlternatingSkip` | Skip every other block (alternating). |
| `MiddleSkip` | Skip blocks in the middle (preserve first and last). |
| `UniformSkip` | Skip blocks at uniform intervals. |

