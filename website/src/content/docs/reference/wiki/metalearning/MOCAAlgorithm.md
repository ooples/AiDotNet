---
title: "MOCAAlgorithm<T, TInput, TOutput>"
description: "Implementation of MOCA: Meta-learning with Online Complementary Augmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MOCA: Meta-learning with Online Complementary Augmentation.

## How It Works

MOCA augments the meta-learning task distribution by generating complementary tasks
in gradient space. For each real task, it creates augmented versions by perturbing the
gradient with a direction orthogonal to the original gradient, scaled by historical
gradient statistics. This explores complementary adaptation directions that improve
generalization and robustness of the meta-learned initialization.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `CreateAugmentedGradient(Vector<>)` | Creates an augmented gradient by adding a complementary perturbation orthogonal to the original gradient direction, scaled by historical variance. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gradMean` | Running mean of gradients across tasks (EMA). |
| `_gradVar` | Running variance of gradients across tasks (EMA). |

