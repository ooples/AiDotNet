---
title: "WeightAnalysisResult<T>"
description: "Analysis results for model weights to guide compression decisions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Analysis results for model weights to guide compression decisions.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusteringPotential` | Gets or sets the clustering potential (how much quantization will help). |
| `Entropy` | Gets or sets the entropy of weight distribution (for Huffman coding potential). |
| `EstimatedCompressionRatio` | Gets or sets the estimated compression ratio achievable. |
| `MaxMagnitude` | Gets or sets the maximum weight magnitude. |
| `MeanMagnitude` | Gets or sets the mean weight magnitude. |
| `MinMagnitude` | Gets or sets the minimum weight magnitude. |
| `NearZeroWeights` | Gets or sets the number of weights with near-zero magnitude. |
| `PruningPotential` | Gets or sets the fraction of weights that are near-zero (pruning potential). |
| `RecommendationReasoning` | Gets or sets the reasoning for the recommendation. |
| `RecommendedParameters` | Gets or sets the recommended hyperparameters for the compression technique. |
| `RecommendedTechnique` | Gets or sets the recommended compression technique based on analysis. |
| `StdDevMagnitude` | Gets or sets the standard deviation of weight magnitudes. |
| `TotalWeights` | Gets or sets the total number of weights analyzed. |
| `UniqueValues` | Gets or sets the number of unique weight values (for clustering potential). |

