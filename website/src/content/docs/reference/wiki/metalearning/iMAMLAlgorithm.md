---
title: "iMAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of the iMAML (Implicit Model-Agnostic Meta-Learning) algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of the iMAML (Implicit Model-Agnostic Meta-Learning) algorithm.

## For Beginners

iMAML solves one of MAML's biggest problems - memory usage.

## How It Works

iMAML is a memory-efficient variant of MAML that uses implicit differentiation to
compute meta-gradients. Instead of backpropagating through all adaptation steps,
it uses the implicit function theorem to directly compute gradients at the adapted
parameters, significantly reducing memory requirements.

Key advantages over MAML:

- Constant memory cost regardless of number of adaptation steps
- Can use many more adaptation steps without memory issues
- Often achieves better performance than first-order MAML (FOMAML)

The problem with MAML:

- To learn from adaptation, MAML needs to remember every step
- More adaptation steps = much more memory needed
- This limits how much adaptation you can do

How iMAML solves it:

- Uses a mathematical shortcut (implicit differentiation)
- Only needs to remember the start and end points
- Can do many more adaptation steps with the same memory

The implicit function theorem allows computing gradients through the adaptation
process by solving: (I + lambda * H)^(-1) * g, where H is the Hessian of the
inner loss and g is the gradient of the query loss. This is solved efficiently
using Conjugate Gradient iteration.

Reference: Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019).
Meta-learning with implicit gradients.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `iMAMLAlgorithm(iMAMLOptions<,,>)` | Initializes a new instance of the iMAMLAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using iMAML's inner loop optimization. |
| `ComputeImplicitMatrixVectorProduct(IFullModel<,,>,IMetaLearningTask<,,>,Vector<>)` | Computes the implicit matrix-vector product (I + lambda * H) * v. |
| `ComputeImplicitMetaGradients(IFullModel<,,>,Vector<>,Vector<>,IMetaLearningTask<,,>)` | Computes implicit meta-gradients using the implicit function theorem. |
| `InnerLoopAdaptation(IFullModel<,,>,IMetaLearningTask<,,>)` | Performs the inner loop adaptation to a specific task. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using iMAML's implicit gradient computation. |
| `SolveImplicitEquation(IFullModel<,,>,IMetaLearningTask<,,>,Vector<>)` | Solves the implicit equation (I + lambda * H) * v = b using Conjugate Gradient. |

