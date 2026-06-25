---
title: "MASTrainerOptions<T>"
description: "Configuration options for the MAS trainer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Trainers`

Configuration options for the MAS trainer.

## For Beginners

These options control how the MAS trainer operates.
MAS protects important parameters by measuring how sensitive the network output
is to each parameter, without needing task labels.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeValidationMetrics` | Whether to compute validation metrics after each epoch. |
| `GradientClipThreshold` | Gradient clipping threshold. |
| `ImportanceBatchSize` | Batch size for importance computation. |
| `ImportanceSamples` | Number of samples to use for importance estimation after task completion. |
| `RegularizationWeight` | Weight for the regularization loss relative to task loss. |
| `ReplayLearningRateFactor` | Fraction of learning rate to use for replay samples. |
| `UseBatchedImportance` | Whether to compute importance in batches for efficiency. |
| `UseExperienceReplay` | Whether to use experience replay alongside MAS regularization. |

