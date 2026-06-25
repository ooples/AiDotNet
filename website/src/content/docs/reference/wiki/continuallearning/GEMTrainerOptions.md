---
title: "GEMTrainerOptions<T>"
description: "Configuration options for the GEM trainer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Trainers`

Configuration options for the GEM trainer.

## For Beginners

These options control how the GEM trainer operates.
GEM prevents forgetting by ensuring gradients don't increase loss on previous tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeValidationMetrics` | Whether to compute validation metrics after each epoch. |
| `ConstraintTolerance` | Tolerance for constraint satisfaction in gradient projection. |
| `ExamplesPerTask` | Number of examples to store per task for gradient computation. |
| `GradientClipThreshold` | Gradient clipping threshold. |
| `MaxQPIterations` | Maximum number of iterations for quadratic programming solver. |
| `MemoryStrength` | Memory strength parameter (gamma) for gradient projection. |
| `ProjectionMargin` | Margin for gradient projection. |
| `UseAveragedGEM` | Whether to use the average gradient (A-GEM) instead of full GEM. |

