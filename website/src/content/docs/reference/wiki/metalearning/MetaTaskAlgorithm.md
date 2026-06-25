---
title: "MetaTaskAlgorithm<T, TInput, TOutput>"
description: "Implementation of MetaTask: Meta-learned Task Augmentation via gradient interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MetaTask: Meta-learned Task Augmentation via gradient interpolation.

## How It Works

MetaTask augments the meta-learning task distribution by generating synthetic tasks
from convex combinations of real task gradients. For each synthetic task, two real tasks
are randomly selected and their gradients are interpolated using a Beta-distributed
coefficient. The meta-objective combines losses from both real and synthetic tasks.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

