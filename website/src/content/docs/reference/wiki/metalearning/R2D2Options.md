---
title: "R2D2Options<T, TInput, TOutput>"
description: "Configuration options for R2-D2 (Meta-learning with Differentiable Closed-form Solvers) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for R2-D2 (Meta-learning with Differentiable Closed-form Solvers) algorithm.

## For Beginners

R2-D2 is like MAML but with a mathematical shortcut:

**The Problem with MAML's inner loop:**

- MAML takes K gradient steps to adapt (iterative, slow)
- Each step requires computing gradients (expensive)
- More steps = more memory for second-order gradients

**R2-D2's solution:**

- Instead of gradient steps, use ridge regression (closed-form, instant)
- Ridge regression has an exact mathematical solution: w = (X^T X + lambda I)^-1 X^T y
- This solution is differentiable, so meta-gradients still flow through it
- Result: Lightning-fast inner loop with one matrix solve instead of K gradient steps

**Analogy:**

- MAML: Walk toward the solution step by step (might need 5-10 steps)
- R2-D2: Jump directly to the solution using math (one step, exact answer)

The catch: R2-D2's "jump" only works for the last layer (linear classifier),
so the feature extractor still needs good meta-learned features.

## How It Works

R2-D2 replaces the iterative inner-loop optimization of MAML with a closed-form differentiable
ridge regression solver. This makes the inner loop extremely fast (single forward pass) while
remaining fully differentiable for meta-gradient computation.

Reference: Bertinetto, L., Henriques, J. F., Torr, P., & Vedaldi, A. (2019).
Meta-learning with Differentiable Closed-form Solvers. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `R2D2Options(IFullModel<,,>)` | Initializes a new instance of the R2D2Options class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps (set to 1 for R2-D2's closed-form solver). |
| `CheckpointFrequency` | Gets or sets checkpoint frequency. |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EmbeddingDimension` | Gets or sets the feature embedding dimension for ridge regression. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints. |
| `EvaluationFrequency` | Gets or sets evaluation frequency. |
| `EvaluationTasks` | Gets or sets evaluation tasks count. |
| `GradientClipThreshold` | Gets or sets the gradient clipping threshold. |
| `InnerLearningRate` | Gets or sets the inner learning rate (not used directly; R2-D2 uses closed-form solver). |
| `InnerOptimizer` | Gets or sets the inner loop optimizer (not used for R2-D2, kept for interface compatibility). |
| `Lambda` | Gets or sets the ridge regression regularization parameter (lambda). |
| `LearnLambda` | Gets or sets whether lambda should be meta-learned. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` | Gets or sets the meta-batch size. |
| `MetaModel` | Gets or sets the feature extractor model (backbone). |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the outer learning rate. |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseWoodburyIdentity` | Gets or sets whether to use Woodbury identity for efficient matrix inversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

