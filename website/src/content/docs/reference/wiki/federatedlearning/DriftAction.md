---
title: "DriftAction"
description: "Recommended action for a drifting client."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.DriftDetection`

Recommended action for a drifting client.

## Fields

| Field | Summary |
|:-----|:--------|
| `Monitor` | Monitor more closely (increase detection frequency). |
| `None` | No action needed. |
| `ReduceWeight` | Reduce aggregation weight for this client. |
| `SelectiveRetrain` | Request selective retraining from this client. |
| `TemporaryExclude` | Exclude this client temporarily until drift stabilizes. |

