---
title: "FilterMode"
description: "Specifies how anomalies should be handled during preprocessing."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.OutlierHandling`

Specifies how anomalies should be handled during preprocessing.

## Fields

| Field | Summary |
|:-----|:--------|
| `Flag` | Add a binary column indicating anomaly status (1 = anomaly, 0 = normal). |
| `Remove` | Remove rows identified as anomalies. |
| `ReplaceWithMean` | Replace anomalous values with the column mean. |
| `ReplaceWithMedian` | Replace anomalous values with the column median. |

