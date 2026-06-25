---
title: "CoresetStrategy"
description: "Strategy for selecting coreset samples."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Quality`

Strategy for selecting coreset samples.

## Fields

| Field | Summary |
|:-----|:--------|
| `Greedy` | Greedy facility location: iteratively pick the point that maximizes coverage. |
| `KCenter` | k-Center: minimize the maximum distance from any point to its nearest selected point. |
| `Random` | Random sampling baseline. |

