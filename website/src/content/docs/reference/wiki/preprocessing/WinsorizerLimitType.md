---
title: "WinsorizerLimitType"
description: "Specifies how Winsorization limits are calculated."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.OutlierHandling`

Specifies how Winsorization limits are calculated.

## Fields

| Field | Summary |
|:-----|:--------|
| `IQR` | Use IQR-based limits (Q1 - k*IQR, Q3 + k*IQR where k is the limit value). |
| `Percentile` | Use percentile values directly (e.g., 5th and 95th percentile). |

