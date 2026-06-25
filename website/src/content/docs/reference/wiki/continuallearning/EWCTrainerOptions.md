---
title: "EWCTrainerOptions<T>"
description: "Configuration options for the EWC trainer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Trainers`

Configuration options for the EWC trainer.

## For Beginners

These options control how the EWC trainer operates.
EWC protects important parameters by adding a regularization penalty when they change.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeValidationMetrics` | Whether to compute validation metrics after each epoch. |
| `GradientClipThreshold` | Gradient clipping threshold. |
| `OnlineEWC` | Whether to accumulate Fisher Information across tasks (online EWC). |
| `RegularizationWeight` | Weight for the regularization loss relative to task loss. |
| `ReplayLearningRateFactor` | Fraction of learning rate to use for replay samples. |
| `UseExperienceReplay` | Whether to use experience replay alongside EWC regularization. |

