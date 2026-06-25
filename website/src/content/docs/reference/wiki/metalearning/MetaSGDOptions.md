---
title: "MetaSGDOptions<T, TInput, TOutput>"
description: "Configuration options for the Meta-SGD (Meta Stochastic Gradient Descent) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the Meta-SGD (Meta Stochastic Gradient Descent) algorithm.

## For Beginners

Think of Meta-SGD as "learning how to learn" at the finest grain:

## How It Works

Meta-SGD extends MAML by learning not just the model initialization but also per-parameter
learning rates, momentum terms, and update directions. This effectively learns a custom
optimizer configuration for each parameter, enabling highly specialized adaptation strategies.

In standard training, you pick one learning rate for all parameters. But different parts
of a neural network might benefit from different learning rates. Meta-SGD figures this out
automatically by learning:

- **Per-parameter learning rates:** Some weights need small updates, others larger
- **Per-parameter momentum:** Some weights benefit from momentum, others don't
- **Update directions:** Sometimes the gradient direction should be flipped or scaled

**Algorithm Overview:**

**Key Insights:**

1. Per-parameter optimization allows heterogeneous learning rates across layers
2. First-order method: no Hessian computation needed, much faster than second-order MAML
3. Learned optimizers reveal which parameters are important for quick adaptation
4. Can combine with various base update rules (SGD, Adam, RMSprop)

**Reference:** Li, Z., Zhou, F., Chen, F., & Li, H. (2017).
Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaSGDOptions(IFullModel<,,>)` | Initializes a new instance of the MetaSGDOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdamBeta1Init` | Gets or sets the initial value for Adam beta1. |
| `AdamBeta2Init` | Gets or sets the initial value for Adam beta2. |
| `AdamEpsilonInit` | Gets or sets the epsilon value for Adam numerical stability. |
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints during training. |
| `EvaluationFrequency` | Gets or sets how often to evaluate during meta-training. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `InnerSteps` | Gets or sets the number of inner steps during meta-training. |
| `LayerDecayFactor` | Gets or sets the decay factor per layer depth. |
| `LearnAdamBetas` | Gets or sets whether to learn Adam beta parameters when using Adam update rule. |
| `LearnDirection` | Gets or sets whether to learn per-parameter update direction signs. |
| `LearnLearningRate` | Gets or sets whether to learn per-parameter learning rates. |
| `LearnMomentum` | Gets or sets whether to learn per-parameter momentum coefficients. |
| `LearningRateInitRange` | Gets or sets the initialization range for learning rates when using random initialization. |
| `LearningRateInitialization` | Gets or sets the initialization strategy for per-parameter learning rates. |
| `LearningRateL2Reg` | Gets or sets the L2 regularization coefficient for learned learning rates. |
| `LearningRateSchedule` | Gets or sets the learning rate schedule type. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MaxLearningRate` | Gets or sets the maximum allowed per-parameter learning rate. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `MinLearningRate` | Gets or sets the minimum allowed per-parameter learning rate. |
| `NumCurvatureSamples` | Gets or sets the number of samples for curvature approximation. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `NumParameterGroups` | Gets or sets the number of parameter groups. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-optimization). |
| `ParameterSharingThreshold` | Gets or sets the similarity threshold for parameter sharing. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `ScheduleWarmupEpisodes` | Gets or sets the number of warmup episodes for learning rate schedule. |
| `TrustRegionRadius` | Gets or sets the trust region radius. |
| `UpdateRuleType` | Gets or sets the update rule type for per-parameter optimization. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseHessianFree` | Gets or sets whether to use Hessian-free approximation for meta-gradients. |
| `UseLayerWiseDecay` | Gets or sets whether to apply layer-wise learning rate decay. |
| `UseLearningRateSchedule` | Gets or sets whether to use a learning rate schedule for meta-learning rates. |
| `UseParameterGrouping` | Gets or sets whether to use parameter grouping. |
| `UseParameterSharing` | Gets or sets whether to use parameter sharing based on similarity. |
| `UseTrustRegion` | Gets or sets whether to use trust region for parameter updates. |
| `UseWarmStart` | Gets or sets whether to use warm-start initialization for the optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the Meta-SGD options. |
| `GetEffectiveParameterGroups(Int32)` | Gets the effective number of parameter groups. |
| `GetTotalMetaParameters(Int32)` | Gets the total number of learnable meta-parameters. |
| `IsValid` | Validates that all Meta-SGD configuration options are properly set. |

