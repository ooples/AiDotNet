---
title: "CAVIAAlgorithm<T, TInput, TOutput>"
description: "Implementation of CAVIA (Fast Context Adaptation via Meta-Learning) for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of CAVIA (Fast Context Adaptation via Meta-Learning) for few-shot learning.

## For Beginners

CAVIA is a smarter version of MAML that adapts faster:

**How it works:**

1. The model has two kinds of parameters:
- Body parameters (shared across all tasks) - these are the model's "core skills"
- Context parameters (adapted per task) - these are task-specific adjustments
2. For a new task, only context parameters are updated (much faster!)
3. The body parameters improve over time across all tasks
4. Context is a small vector concatenated with/added to the input

**Simple example:**

- Body params: How to recognize shapes, edges, textures (shared knowledge)
- Context params: "Right now I'm looking at animals" vs "Right now I'm looking at vehicles"
- For each new task, only adjust what kind of thing you're looking at
- Your fundamental perception skills stay the same

**Why it's better than MAML:**

- MAML adapts ALL parameters (expensive, O(P) where P = total params)
- CAVIA adapts only context (cheap, O(C) where C = context_dim, C << P)
- Less prone to meta-overfitting (fewer adapted parameters)
- Mathematically cleaner separation of shared vs. task-specific knowledge

## How It Works

CAVIA separates model parameters into shared body parameters and task-specific context
parameters. Only the small context vector is adapted during the inner loop, making CAVIA
significantly faster than full MAML while achieving comparable performance.

**Algorithm - CAVIA:**

**Key Insights:**

1. **Separation of Concerns**: Body parameters capture shared structure across tasks,

while context parameters capture task-specific information.

2. **Efficient Adaptation**: Only adapting the small context vector means the inner

loop is O(context_dim) instead of O(total_params), typically 100x-1000x cheaper.

3. **Reduced Meta-Overfitting**: Fewer adapted parameters means less risk of overfitting

the meta-learner to the training tasks.

4. **Interpretable Context**: The context vector provides a low-dimensional representation

of what makes each task unique.

Reference: Zintgraf, L. M., Shiarli, K., Kurin, V., Hofmann, K., & Whiteson, S. (2019).
Fast Context Adaptation via Meta-Learning. ICML 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CAVIAAlgorithm(CAVIAOptions<,,>)` | Initializes a new instance of the CAVIAAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task by learning task-specific context parameters. |
| `AdaptContext(Vector<>,,)` | Adapts the context vector using gradient descent on the support set. |
| `AugmentInput(,Vector<>)` | Augments the input by injecting the context vector according to the configured injection mode. |
| `ComputeContextGradients(Vector<>,,)` | Computes gradients of the loss with respect to context parameters using finite differences. |
| `CreateInitialContext` | Creates the initial context vector with the configured initial value. |
| `GetOptions` |  |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using CAVIA's context adaptation strategy. |

