---
title: "MAMLPlusPlusOptions<T, TInput, TOutput>"
description: "Configuration options for MAML++ (How to Train Your MAML) meta-learning algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for MAML++ (How to Train Your MAML) meta-learning algorithm.

## For Beginners

MAML++ is "MAML done right" for production use.

Original MAML has several practical problems:

- Training can be unstable (loss explodes)
- Batch normalization doesn't work well
- A single learning rate isn't optimal for all inner steps
- Second-order gradients are expensive and often unhelpful early on

MAML++ fixes ALL of these with:

1. **Multi-Step Loss**: Supervise every inner step, not just the last one
2. **Per-Step Learning Rates**: Each adaptation step has its own learning rate
3. **Derivative-Order Annealing**: Start with first-order, gradually add second-order
4. **Stable Batch Norm**: Per-step batch norm statistics for stable adaptation

Think of it like upgrading from a prototype to production code - same idea, but
with all the engineering needed to work reliably at scale.

## How It Works

MAML++ is a production-hardened version of MAML that addresses several training instabilities
through multi-step loss optimization, derivative-order annealing, per-step learning rates,
and batch normalization stability fixes.

Reference: Antoniou, A., Edwards, H., & Storkey, A. (2019).
How to Train Your MAML. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAMLPlusPlusOptions(IFullModel<,,>)` | Initializes a new instance of the MAMLPlusPlusOptions class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of inner-loop adaptation steps. |
| `AnnealingIterations` | Gets or sets the number of iterations over which to anneal from first-order to second-order. |
| `CheckpointFrequency` | Gets or sets the checkpoint frequency. |
| `CosineAnnealingMinRatio` | Gets or sets the cosine annealing schedule minimum learning rate ratio. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints. |
| `EvaluationFrequency` | Gets or sets the evaluation frequency. |
| `EvaluationTasks` | Gets or sets the number of evaluation tasks. |
| `GradientClipThreshold` | Gets or sets the gradient clipping threshold. |
| `InnerLearningRate` | Gets or sets the base inner learning rate (used for initialization of per-step rates). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks per meta-batch. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for outer loop updates. |
| `MultiStepLossWeights` | Gets or sets the weights for multi-step loss at each adaptation step. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the outer learning rate for meta-parameter updates. |
| `RandomSeed` | Gets or sets the random seed. |
| `UseDerivativeOrderAnnealing` | Gets or sets whether to use derivative-order annealing. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseMultiStepLoss` | Gets or sets whether to use multi-step loss optimization (MSL). |
| `UsePerStepBatchNorm` | Gets or sets whether to use per-step batch normalization statistics. |
| `UsePerStepLearningRates` | Gets or sets whether to use per-step learning rates (LSLR). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

