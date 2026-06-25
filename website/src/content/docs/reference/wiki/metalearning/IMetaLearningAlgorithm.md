---
title: "IMetaLearningAlgorithm<T, TInput, TOutput>"
description: "IMetaLearningAlgorithm<T, TInput, TOutput> — Interfaces in AiDotNet.MetaLearning.Algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.MetaLearning.Algorithms`

_No summary documentation available yet._

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets the number of adaptation steps to perform during task adaptation. |
| `AlgorithmName` | Gets the name of the meta-learning algorithm. |
| `InnerLearningRate` | Gets the learning rate used for task adaptation (inner loop). |
| `OuterLearningRate` | Gets the learning rate used for meta-learning (outer loop). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the model to a new task using its support set. |
| `Evaluate(TaskBatch<,,>)` | Evaluates the meta-learning algorithm on a batch of tasks. |
| `GetMetaModel` | Gets the base model used by this meta-learning algorithm. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step on a batch of tasks. |
| `SetMetaModel(IFullModel<,,>)` | Sets the base model for this meta-learning algorithm. |

