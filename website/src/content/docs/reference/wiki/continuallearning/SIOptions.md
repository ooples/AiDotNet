---
title: "SIOptions<T>"
description: "Configuration options for Synaptic Intelligence."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Strategies`

Configuration options for Synaptic Intelligence.

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulationMode` | Gets or sets the importance accumulation mode. |
| `Damping` | Gets or sets the damping constant (xi in the paper). |
| `DecayFactor` | Gets or sets the decay factor for weighted accumulation. |
| `Lambda` | Gets or sets the regularization strength (c in the paper). |
| `MinImportanceValue` | Gets or sets the minimum importance value to prevent underflow. |
| `NormalizeImportance` | Gets or sets whether to normalize importance values. |
| `TrackLayerStatistics` | Gets or sets whether to track per-layer importance statistics. |
| `UseRunningAverage` | Gets or sets whether to use running average for path integral. |

