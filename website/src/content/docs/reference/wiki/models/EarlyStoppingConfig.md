---
title: "EarlyStoppingConfig"
description: "Early stopping configuration for a training stage."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Early stopping configuration for a training stage.

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets whether early stopping is enabled. |
| `LowerIsBetter` | Gets or sets whether lower values are better (true for loss, false for accuracy). |
| `MinDelta` | Gets or sets the minimum change to qualify as an improvement. |
| `MonitorMetric` | Gets or sets the metric to monitor for early stopping. |
| `Patience` | Gets or sets the number of epochs with no improvement before stopping. |

