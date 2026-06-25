---
title: "SITrainerOptions<T>"
description: "Configuration options for the SI trainer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Trainers`

Configuration options for the SI trainer.

## For Beginners

These options control how the SI trainer operates.
SI tracks parameter importance along the optimization trajectory.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeValidationMetrics` | Whether to compute validation metrics after each epoch. |
| `DampingParameter` | Damping parameter (ξ) to prevent division by zero in importance computation. |
| `GradientClipThreshold` | Gradient clipping threshold. |
| `NormalizeImportance` | Whether to normalize importance values after each task. |
| `PathIntegralDecay` | Decay factor for path integral accumulation. |
| `RegularizationWeight` | Weight for the regularization loss relative to task loss. |
| `ReplayLearningRateFactor` | Fraction of learning rate to use for replay samples. |
| `UseExperienceReplay` | Whether to use experience replay alongside SI regularization. |

