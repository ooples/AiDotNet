---
title: "CompressionOptimizerOptions"
description: "Configuration options for the compression optimizer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.AutoML`

Configuration options for the compression optimizer.

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyWeight` | Gets or sets the weight for accuracy in fitness calculation (default: 0.5). |
| `CompressionWeight` | Gets or sets the weight for compression ratio in fitness calculation (default: 0.3). |
| `IncludeEncoding` | Gets or sets whether to include encoding techniques (default: true). |
| `IncludeHybrid` | Gets or sets whether to include hybrid techniques like Deep Compression (default: true). |
| `IncludePruning` | Gets or sets whether to include pruning techniques (default: true). |
| `IncludeQuantization` | Gets or sets whether to include quantization techniques (default: true). |
| `MaxAccuracyLoss` | Gets or sets the maximum acceptable accuracy loss as a fraction (default: 0.02 = 2%). |
| `MaxTrials` | Gets or sets the maximum number of trials to run (default: 20). |
| `MinCompressionRatio` | Gets or sets the minimum acceptable compression ratio (default: 2.0). |
| `RandomSeed` | Gets or sets the random seed for reproducibility (default: null for random). |
| `SpeedWeight` | Gets or sets the weight for inference speed in fitness calculation (default: 0.2). |

