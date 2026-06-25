---
title: "EdgeHandling"
description: "How to handle edge cases where the full window is not available."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

How to handle edge cases where the full window is not available.

## Fields

| Field | Summary |
|:-----|:--------|
| `ForwardFill` | Use the first available value to fill the beginning. |
| `NaN` | Fill with NaN where window extends beyond data boundaries. |
| `Partial` | Use partial windows (calculate with available data). |
| `Truncate` | Truncate output to only include complete windows. |

