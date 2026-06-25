---
title: "AdamAnomalyGuardMode"
description: "Policy for the PyTorch GradScaler-style anomaly guard on the Adam optimizer's tape-based `Step`."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Policy for the PyTorch GradScaler-style anomaly guard on the Adam
optimizer's tape-based `Step`. The guard scans every gradient
element for NaN/Inf and skips the entire step when one is found,
preventing permanent poisoning of the `m`/`v` moment
accumulators.

## Fields

| Field | Summary |
|:-----|:--------|
| `Always` | Always scan gradients before each step. |
| `Auto` | Default: scan gradients before each step. |
| `Never` | Skip the anomaly scan entirely. |

