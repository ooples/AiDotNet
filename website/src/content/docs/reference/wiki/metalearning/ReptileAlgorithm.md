---
title: "ReptileAlgorithm<T, TInput, TOutput>"
description: "Implementation of the Reptile meta-learning algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of the Reptile meta-learning algorithm.

## For Beginners

Reptile is like learning by averaging your experiences.

## How It Works

Reptile is a simple and scalable meta-learning algorithm. Unlike MAML, it doesn't require
computing gradients through the adaptation process, making it more efficient and easier
to implement while achieving competitive performance.

**Algorithm:**

1. Sample a task (or batch of tasks)
2. Perform SGD on the task starting from the current meta-parameters
3. Update meta-parameters by interpolating toward the adapted parameters
4. Repeat

Imagine learning to cook:

- You start with basic knowledge (initial parameters)
- You make a specific dish and learn specific techniques
- Instead of just remembering that one dish, you update your basic knowledge

to include some of what you learned

- After cooking many dishes, your basic knowledge becomes really good

for learning any new recipe quickly

Reptile is simpler than MAML because it just moves toward adapted parameters
instead of computing complex gradients through the adaptation process.
The key insight is that this simple approach achieves similar performance
to more complex methods like MAML.

Reference: Nichol, A., Achiam, J., & Schulman, J. (2018).
On first-order meta-learning algorithms.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReptileAlgorithm(ReptileOptions<,,>)` | Initializes a new instance of the ReptileAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using gradient descent. |
| `InnerLoopAdaptation(IFullModel<,,>,IMetaLearningTask<,,>)` | Performs the inner loop adaptation to a specific task. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using Reptile's parameter interpolation approach. |

