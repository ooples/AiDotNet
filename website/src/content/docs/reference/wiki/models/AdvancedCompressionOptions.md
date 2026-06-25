---
title: "AdvancedCompressionOptions"
description: "Configuration for advanced gradient compression methods (PowerSGD, sketching, adaptive)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for advanced gradient compression methods (PowerSGD, sketching, adaptive).

## For Beginners

These options configure state-of-the-art compression methods
that can reduce communication bandwidth by 100-1000x compared to uncompressed gradients.
Set this on `FederatedCompressionOptions` when using advanced compression.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptiveBandwidthWindow` | Gets or sets the number of recent rounds used for bandwidth estimation in adaptive mode. |
| `AdaptiveMaxRatio` | Gets or sets the maximum compression ratio for adaptive mode (ceiling). |
| `AdaptiveMinRatio` | Gets or sets the minimum compression ratio for adaptive mode (floor). |
| `FedDTMaxDepth` | Gets or sets the maximum tree depth for FedDT decision-tree compression. |
| `FedKDTemperature` | Gets or sets the knowledge distillation temperature for FedKD compression. |
| `FetchSGDTopK` | Gets or sets the number of top-K heavy hitters to recover from FetchSGD sketches. |
| `PowerSGDRank` | Gets or sets the rank for PowerSGD low-rank approximation. |
| `PowerSGDWarmStart` | Gets or sets whether PowerSGD uses warm-start (reuses previous round's factors). |
| `SignSGDLearningRate` | Gets or sets the learning rate for SignSGD compression. |
| `SketchDepth` | Gets or sets the number of hash functions for Count Sketch compression. |
| `SketchTopK` | Gets or sets the number of top-k elements to recover from a sketch. |
| `SketchWidth` | Gets or sets the sketch width (number of buckets per hash function). |
| `Strategy` | Gets or sets the advanced compression strategy. |
| `UseErrorFeedback` | Gets or sets whether to use error feedback (residual accumulation) with any compressor. |

