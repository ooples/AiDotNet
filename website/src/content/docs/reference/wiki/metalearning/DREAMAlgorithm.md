---
title: "DREAMAlgorithm<T, TInput, TOutput>"
description: "Implementation of DREAM: Directed REward Augmented Meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of DREAM: Directed REward Augmented Meta-learning.

## How It Works

DREAM meta-learns a reward shaping function that transforms the raw task loss into a more
informative signal for the inner loop. The reward shaper is a small MLP that takes
(loss, gradient_norm, step/K) as input and outputs a scalar multiplier for the gradient.
This enables curriculum-like adaptation where early steps may be scaled differently
than later steps.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeRewardScale(Double,Double,Double)` | Computes the reward scaling factor using the shaper MLP: input = (loss, grad_norm, progress) → hidden(tanh) → sigmoid(output) |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_shaperParams` | Reward shaper parameters: 3-input → hidden → 1-output MLP. |

