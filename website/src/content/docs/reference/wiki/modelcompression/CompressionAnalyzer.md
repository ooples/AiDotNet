---
title: "CompressionAnalyzer<T>"
description: "Analyzes model weights to determine optimal compression strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Analyzes model weights to determine optimal compression strategies.

## For Beginners

Before compressing a model, it helps to understand its weights.

This analyzer looks at your model's weights and answers questions like:

- Are many weights already close to zero? (Good for pruning)
- Are weights clustered around certain values? (Good for quantization)
- What's the distribution of weight values? (Affects all techniques)

Based on this analysis, it recommends:

- Which compression technique to use
- What settings (hyperparameters) to use
- What compression ratio to expect

This helps you make informed decisions without trial-and-error.

## How It Works

CompressionAnalyzer examines model weight distributions to recommend the best compression
technique and hyperparameters. It analyzes properties like weight sparsity, magnitude
distribution, and redundancy to make informed recommendations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompressionAnalyzer(Double,Int32)` | Initializes a new instance of the CompressionAnalyzer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Analyze(Vector<>,Boolean)` | Analyzes model weights and returns compression recommendations. |
| `CalculateEntropy(Double[])` | Calculates the entropy of the weight distribution. |
| `EstimateUniqueValues(Vector<>)` | Estimates the number of unique values using sampling for large arrays. |
| `GenerateReport(WeightAnalysisResult<>)` | Generates a detailed analysis report. |
| `MakeRecommendation(WeightAnalysisResult<>,Boolean)` | Makes compression recommendations based on analysis results. |

