---
title: "WarpGradOptions<T, TInput, TOutput>"
description: "Configuration options for the WarpGrad (Warped Gradient Descent) meta-learning algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the WarpGrad (Warped Gradient Descent) meta-learning algorithm.

## For Beginners

WarpGrad is like learning a better map for navigating a landscape.

Imagine you're trying to reach the bottom of a valley (optimal solution):

- Standard gradient descent: Walk straight downhill (may zigzag on narrow valleys)
- MAML: Start from a better position on the hillside
- WarpGrad: Learn to reshape the landscape so downhill is always the right direction

The "warp layers" transform the gradients (direction signals) so that even simple
gradient descent moves efficiently toward the solution. This is like giving the model
a compass that's been calibrated for different types of tasks.

Key advantages over MAML:

- No need to backpropagate through the inner loop (much cheaper)
- Warp layers provide task-independent gradient conditioning
- Can be combined with any inner-loop optimizer

## How It Works

WarpGrad learns a preconditioning matrix (warp-layers) that transforms gradients during
inner-loop adaptation. Unlike MAML which learns a good initialization, WarpGrad learns
a good gradient descent geometry that makes adaptation more efficient.

Reference: Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., & Hadsell, R. (2020).
Meta-Learning with Warped Gradient Descent. ICLR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WarpGradOptions(IFullModel<,,>)` | Initializes a new instance of the WarpGradOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps for inner loop adaptation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task-specific adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates (task parameter updates). |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for outer loop updates (warp-layer and shared parameter updates). |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `NumWarpLayers` | Gets or sets the number of warp-layers to interleave with the task learner. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-parameter and warp-layer updates). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseDiagonalWarp` | Gets or sets whether warp-layers use diagonal (element-wise) or full matrix transformations. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation for meta-gradients. |
| `WarpInitScale` | Gets or sets the initialization scale for warp-layer parameters. |
| `WarpLayerHiddenDim` | Gets or sets the hidden dimension for warp-layer MLPs. |
| `WarpLearningRate` | Gets or sets the learning rate for warp-layer parameter updates. |
| `WarpRegularization` | Gets or sets the L2 regularization weight for warp-layer parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a copy of the WarpGrad options. |
| `IsValid` | Validates that all WarpGrad configuration options are properly set. |

