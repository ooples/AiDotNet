---
title: "MetaOptNetOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm.

## For Beginners

In MAML, the inner loop does gradient descent:
"take a step, take a step, take a step..." which can be slow and unstable.
MetaOptNet says: "Why iterate? Just solve the optimal answer directly!"

## How It Works

MetaOptNet replaces the gradient-based inner-loop of MAML with a differentiable
convex optimization solver. Instead of taking gradient steps, it solves a closed-form
optimization problem (like ridge regression or SVM) to get the classifier.

It uses mathematical formulas that give the best classifier in one shot.
This is:

- **Faster:** No iterative optimization
- **More stable:** Convex problems have unique solutions
- **Theoretically grounded:** Based on well-understood optimization theory

Reference: Lee, K., Maji, S., Ravichandran, A., & Soatto, S. (2019).
Meta-Learning with Differentiable Convex Optimization. CVPR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaOptNetOptions(IFullModel<,,>)` | Initializes a new instance of the MetaOptNetOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EmbeddingDimension` | Gets or sets the dimension of the feature embedding. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EncoderL2Regularization` | Gets or sets the L2 regularization strength for the encoder. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InitialTemperature` | Gets or sets the initial temperature value. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (not used - solver is analytical). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MaxSolverIterations` | Gets or sets the maximum number of iterations for iterative solvers (like SVM). |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (feature encoder) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter (outer loop) updates. |
| `NormalizeEmbeddings` | Gets or sets whether to normalize embeddings before solving. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (encoder training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RegularizationStrength` | Gets or sets the regularization parameter for the convex solver. |
| `SolverTolerance` | Gets or sets the convergence tolerance for iterative solvers. |
| `SolverType` | Gets or sets the type of convex solver to use. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseLearnedTemperature` | Gets or sets whether to use a learned temperature for scaling. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the MetaOptNet options. |
| `IsValid` | Validates that all MetaOptNet configuration options are properly set. |

