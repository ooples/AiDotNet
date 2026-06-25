---
title: "CompressionMetrics<T>"
description: "Provides metrics and statistics for model compression operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Provides metrics and statistics for model compression operations.

## For Beginners

CompressionMetrics helps you measure how well compression worked.

When you compress a model, you want to know:

- How much smaller did it get? (compression ratio)
- How much memory did we save? (size reduction)
- Did it get faster? (inference speed)
- Is it still accurate? (accuracy preservation)

This class tracks all these important measurements so you can:

- Compare different compression techniques
- Decide if the compression is worth it
- Find the best balance between size and accuracy

Example:

- Original model: 100 MB, 95% accuracy, 10ms inference
- Compressed model: 10 MB, 94% accuracy, 5ms inference
- Metrics show: 10x compression, 1% accuracy loss, 2x speedup
- Conclusion: Great compression! The small accuracy loss is worth the huge size reduction.

## How It Works

CompressionMetrics tracks important statistics about the compression process, including
compression ratio, model size reduction, inference speed impact, and accuracy preservation.
These metrics help evaluate the effectiveness of different compression strategies.

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyLoss` | Gets or sets the accuracy loss percentage. |
| `BitsPerWeight` | Gets or sets the number of bits per weight after quantization. |
| `CompressedAccuracy` | Gets or sets the compressed model accuracy (after compression). |
| `CompressedInferenceTimeMs` | Gets or sets the compressed model inference time in milliseconds. |
| `CompressedSize` | Gets or sets the compressed model size in bytes. |
| `CompressionRatio` | Gets or sets the compression ratio (original size / compressed size). |
| `CompressionTechnique` | Gets or sets the compression technique used. |
| `CompressionTimeMs` | Gets or sets the time taken to perform compression in milliseconds. |
| `DecompressionTimeMs` | Gets or sets the time taken to decompress in milliseconds. |
| `EffectiveParameterCount` | Gets or sets the effective number of unique parameters after compression. |
| `InferenceSpeedup` | Gets or sets the inference speedup factor. |
| `MemoryBandwidthSavings` | Gets or sets the memory bandwidth savings ratio. |
| `OriginalAccuracy` | Gets or sets the original model accuracy (before compression). |
| `OriginalInferenceTimeMs` | Gets or sets the original inference time in milliseconds. |
| `OriginalParameterCount` | Gets or sets the number of parameters in the original model. |
| `OriginalSize` | Gets or sets the original model size in bytes. |
| `ReconstructionError` | Gets or sets the reconstruction error (for lossy compression). |
| `SizeReductionPercentage` | Gets or sets the percentage of size reduction. |
| `Sparsity` | Gets or sets the sparsity level achieved (fraction of zero weights). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCompositeFitness(Double,Double,Double)` | Calculates a composite fitness score for multi-objective optimization. |
| `CalculateDerivedMetrics` | Calculates all derived metrics from the base measurements. |
| `FormatBytes(Int64)` | Formats a byte count into a human-readable string. |
| `FromDeepCompressionStats(DeepCompressionStats,String)` | Creates a CompressionMetrics instance from a DeepCompressionStats object. |
| `IsBetterThan(CompressionMetrics<>,Double,Double,Double)` | Compares this compression result to another and determines which is better. |
| `MeetsQualityThreshold(,)` | Determines if the compression meets the specified quality threshold. |
| `MeetsQualityThreshold(Double,Double)` | Determines if the compression meets the specified quality threshold using default values. |
| `ToString` | Gets a human-readable summary of the compression metrics. |

