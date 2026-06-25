---
title: "DriftType"
description: "Drift classification for a client's data distribution."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.DriftDetection`

Drift classification for a client's data distribution.

## Fields

| Field | Summary |
|:-----|:--------|
| `Gradual` | Gradual drift: slow transition between concepts. |
| `None` | No drift detected. |
| `Recurring` | Recurring drift: previously seen distribution returning. |
| `Sudden` | Sudden drift: abrupt change in distribution. |
| `Warning` | Warning: drift may be imminent. |

