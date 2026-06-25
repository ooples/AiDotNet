---
title: "ThresholdMode"
description: "Threshold comparison mode."
section: "API Reference"
---

`Enums` · `AiDotNet.LearningRateSchedulers`

Threshold comparison mode.

## Fields

| Field | Summary |
|:-----|:--------|
| `Absolute` | Static threshold: best + threshold for max, best - threshold for min |
| `Relative` | Dynamic threshold: best * (1 + threshold) for max, best * (1 - threshold) for min |

