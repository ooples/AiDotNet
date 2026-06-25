---
title: "GEMOptions<T>"
description: "Configuration options for Gradient Episodic Memory strategies."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Strategies`

Configuration options for Gradient Episodic Memory strategies.

## For Beginners

These options control how GEM prevents forgetting by
constraining gradients to not harm previous task performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `GradientBatchSize` | Batch size for gradient computation from memory. |
| `Margin` | Margin/epsilon for constraint violations. |
| `MaxQPIterations` | Maximum number of iterations for the QP solver. |
| `MemorySamplingStrategy` | Memory sampling strategy for storing examples. |
| `MemorySizePerTask` | Number of examples to store per task. |
| `NormalizeGradients` | Whether to normalize gradients before checking constraints. |
| `QPTolerance` | Tolerance for QP solver convergence. |
| `RandomSeed` | Random seed for reproducibility. |
| `ReplaySamplingStrategy` | Replay sampling strategy for gradient computation. |
| `UseAGEM` | Whether to use Averaged GEM (A-GEM) instead of full GEM. |

