---
title: "iMAMLOptions<T, TInput, TOutput>"
description: "Configuration options for iMAML (Implicit Model-Agnostic Meta-Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for iMAML (Implicit Model-Agnostic Meta-Learning) algorithm.

## For Beginners

iMAML is a more memory-efficient version of MAML.
Regular MAML needs to remember every adaptation step (expensive!).
iMAML uses a mathematical trick to get the same result with constant memory.

## How It Works

iMAML extends MAML by using implicit differentiation to compute meta-gradients,
which allows for many more adaptation steps without increased memory usage.

Key parameters specific to iMAML:

- LambdaRegularization: Controls stability of implicit gradient computation
- ConjugateGradientIterations: How many CG steps to solve the implicit equation
- ConjugateGradientTolerance: When to stop CG early if converged

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `iMAMLOptions(IFullModel<,,>)` | Initializes a new instance of the iMAMLOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `ConjugateGradientIterations` | Gets or sets the maximum number of Conjugate Gradient iterations. |
| `ConjugateGradientTolerance` | Gets or sets the convergence tolerance for Conjugate Gradient. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints during training. |
| `EvaluationFrequency` | Gets or sets how often to evaluate during meta-training. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `LambdaRegularization` | Gets or sets the regularization strength for implicit gradient computation. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `NeumannSeriesTerms` | Gets or sets the number of terms in the Neumann series approximation. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-optimization). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseNeumannApproximation` | Gets or sets whether to use the Neumann series approximation for implicit gradients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the iMAML options. |
| `IsValid` | Validates that all iMAML configuration options are properly set. |

