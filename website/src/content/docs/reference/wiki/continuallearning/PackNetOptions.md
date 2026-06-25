---
title: "PackNetOptions<T>"
description: "Configuration options for PackNet strategy."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Strategies`

Configuration options for PackNet strategy.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowPrunedReuse` | Gets or sets whether to allow retraining of pruned parameters. |
| `FineTuningEpochs` | Gets or sets the number of fine-tuning epochs after pruning. |
| `LayerWisePruning` | Gets or sets whether to use layer-wise pruning ratios. |
| `MinWeightMagnitude` | Gets or sets the minimum weight magnitude to keep (absolute value). |
| `PruningRatio` | Gets or sets the pruning ratio (fraction of parameters to prune after each task). |
| `UseMagnitudePruning` | Gets or sets whether to use magnitude-based pruning. |

