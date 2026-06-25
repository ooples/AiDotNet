---
title: "MetaSGDAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-SGD (Meta Stochastic Gradient Descent) algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-SGD (Meta Stochastic Gradient Descent) algorithm.

## For Beginners

Meta-SGD learns how to update each parameter individually:

## How It Works

Meta-SGD learns per-parameter learning rates for meta-learning. Instead of
learning just initialization parameters like MAML, it learns the learning
rate, momentum, and direction for each parameter individually, which can be
seen as learning a custom optimizer for each parameter.

In regular training, you use one learning rate for all weights. But different
parts of a neural network benefit from different learning rates. Meta-SGD
figures this out automatically by learning:

- **α_i:** The optimal learning rate for parameter i
- **β_i:** The optimal momentum for parameter i (optional)
- **d_i:** The optimal update direction/sign for parameter i (optional)

**Algorithm - Meta-SGD:**

**Key Insights:**

1. **Per-Parameter Optimization:** Each parameter gets its own learned

optimizer configuration, allowing heterogeneous learning rates across layers.

2. **First-Order Method:** No Hessian computation needed, much faster than

second-order MAML while maintaining strong performance.

3. **Interpretable:** Learned per-parameter learning rates reveal which

parameters are most important for quick adaptation.

4. **Flexible Update Rules:** Can combine with various base optimizers

(SGD, Adam, RMSprop) for different adaptation characteristics.

**Reference:** Li, Z., Zhou, F., Chen, F., & Li, H. (2017).
Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaSGDAlgorithm(MetaSGDOptions<,,>)` | Initializes a new instance of the MetaSGDAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using the learned per-parameter optimizers. |
| `AdaptWithLearnedOptimizer(IMetaLearningTask<,,>,IFullModel<,,>,PerParameterOptimizer<,,>)` | Performs adaptation using the learned per-parameter optimizer. |
| `ComputeMetaGradients(Vector<>,Vector<>,,,,,)` | Computes meta-gradients for updating the per-parameter optimizer coefficients. |
| `ComputePerturbedLoss(Int32,Double,Vector<>,,,,)` | Computes the loss with a perturbed learning rate for finite difference gradient. |
| `GetInitialLearningRate(Int32,Int32)` | Gets the initial learning rate for a parameter based on the initialization strategy. |
| `GetOptions` |  |
| `InitializeOptimizer` | Initializes the per-parameter optimizer with warm-start values. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using Meta-SGD's per-parameter optimization approach. |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains the model and per-parameter optimizer on a single episode. |

