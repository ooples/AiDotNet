---
title: "MOCAOptions<T, TInput, TOutput>"
description: "Configuration options for the MOCA (Meta-learning with Online Complementary Augmentation) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the MOCA (Meta-learning with Online Complementary Augmentation) algorithm.

## How It Works

MOCA augments tasks in gradient space using complementary perturbations derived from
historical gradient statistics. The augmented gradients explore directions orthogonal
to the original task gradient, encouraging more robust meta-learned initializations.

## Properties

| Property | Summary |
|:-----|:--------|
| `AugmentationStrength` | Magnitude of gradient-space perturbation for augmented tasks. |
| `ComplementaryWeight` | Weight on the augmented task loss in the meta-objective. |
| `HistoryMomentum` | EMA momentum for accumulating gradient history statistics (mean and variance). |
| `NumAugmentedTasks` | Number of augmented task variants generated per real task in the meta-batch. |

