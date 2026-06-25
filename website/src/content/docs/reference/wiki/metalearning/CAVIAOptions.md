---
title: "CAVIAOptions<T, TInput, TOutput>"
description: "Configuration options for the CAVIA (Fast Context Adaptation via Meta-Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the CAVIA (Fast Context Adaptation via Meta-Learning) algorithm.

## For Beginners

CAVIA is like MAML but smarter about what it adapts.

Imagine you're a chef learning to cook different cuisines:

- Your cooking skills (shared params) stay the same across cuisines
- But for each cuisine, you adjust your seasoning preferences (context params)
- CAVIA only adjusts the "seasoning" for each new task, not all your skills
- This makes adaptation much faster and less likely to overfit

The context vector is a small set of numbers that tells the model
"this is the kind of task we're dealing with." During adaptation,
only these numbers change - the main model stays fixed.

## How It Works

CAVIA separates model parameters into two groups:

1. **Shared parameters (body):** Updated only in the outer loop across all tasks
2. **Context parameters:** Task-specific, adapted in the inner loop per task

By only adapting context parameters in the inner loop, CAVIA is:

- Much faster than MAML (fewer parameters to differentiate through)
- Less prone to meta-overfitting (fewer adapted parameters = stronger regularization)
- Conceptually cleaner (explicit separation of shared vs. task-specific knowledge)

Reference: Zintgraf, L. M., Shiarli, K., Kurin, V., Hofmann, K., & Whiteson, S. (2019).
Fast Context Adaptation via Meta-Learning. ICML 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CAVIAOptions(IFullModel<,,>)` | Initializes a new instance of the CAVIAOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps for inner loop context adaptation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `ContextDimension` | Gets or sets the dimension of the context parameter vector. |
| `ContextInitValue` | Gets or sets the initial value for context parameters at the start of each task. |
| `ContextInjectionMode` | Gets or sets how the context vector is injected into the model's computation. |
| `ContextRegularizationStrength` | Gets or sets the L2 regularization strength for context parameters. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (context parameter adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates (context parameter updates). |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for outer loop updates (shared parameter updates). |
| `NumContextVectors` | Gets or sets the number of context vectors to use. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (shared parameter updates). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseContextRegularization` | Gets or sets whether to apply L2 regularization on context parameters during adaptation. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation for meta-gradients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the CAVIA options. |
| `IsValid` | Validates that all CAVIA configuration options are properly set. |

