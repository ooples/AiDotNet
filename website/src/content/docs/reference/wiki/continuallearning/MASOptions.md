---
title: "MASOptions<T>"
description: "Configuration options for Memory Aware Synapses."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Strategies`

Configuration options for Memory Aware Synapses.

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulationMode` | Gets or sets the importance accumulation mode across tasks. |
| `BatchSize` | Gets or sets the batch size for importance computation. |
| `DecayFactor` | Gets or sets the decay factor for weighted accumulation. |
| `ImportanceMode` | Gets or sets the importance computation mode. |
| `Lambda` | Gets or sets the regularization strength (lambda). |
| `MinImportanceValue` | Gets or sets the minimum importance value to prevent underflow. |
| `NormalizeImportance` | Gets or sets whether to normalize importance values. |
| `NumSamples` | Gets or sets the number of samples to use for importance estimation. |
| `UseBatching` | Gets or sets whether to use mini-batching for importance computation. |
| `UseL1Norm` | Gets or sets whether to use L1 norm instead of L2 for output sensitivity. |

