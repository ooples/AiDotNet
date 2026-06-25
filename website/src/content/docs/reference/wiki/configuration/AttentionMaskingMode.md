---
title: "AttentionMaskingMode"
description: "Controls how attention masking is applied for optimized attention implementations."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Controls how attention masking is applied for optimized attention implementations.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically select masking based on model/task heuristics. |
| `Causal` | Apply causal masking (autoregressive decoding). |
| `Disabled` | Do not apply causal masking. |

