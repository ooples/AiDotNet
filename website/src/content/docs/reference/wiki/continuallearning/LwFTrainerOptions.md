---
title: "LwFTrainerOptions<T>"
description: "Configuration options for the LwF trainer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Trainers`

Configuration options for the LwF trainer.

## For Beginners

These options control how the LwF trainer operates.
LwF uses knowledge distillation to preserve knowledge from previous tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeValidationMetrics` | Whether to compute validation metrics after each epoch. |
| `DistillationWarmupEpochs` | Warmup epochs before applying full distillation weight. |
| `DistillationWeight` | Weight for the distillation loss relative to task loss. |
| `GradientClipThreshold` | Gradient clipping threshold. |
| `IncludeDistillationInGradients` | Whether to include distillation loss in gradient updates (true) or only use it for monitoring (false). |
| `ReplayLearningRateFactor` | Fraction of learning rate to use for replay samples. |
| `Temperature` | Temperature for softmax in distillation. |
| `UseExperienceReplay` | Whether to use experience replay alongside distillation. |

