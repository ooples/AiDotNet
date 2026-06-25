---
title: "ProximalGradientDescentOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Proximal Gradient Descent optimizer, an advanced optimization algorithm that combines traditional gradient descent with proximal operators to handle regularization effectively."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Proximal Gradient Descent optimizer, an advanced optimization algorithm
that combines traditional gradient descent with proximal operators to handle regularization effectively.

## For Beginners

Proximal Gradient Descent is a specialized optimization method that helps train machine learning models with regularization.

Imagine you're trying to find the lowest point in a hilly landscape while also staying within certain boundaries:

- Regular gradient descent is like always walking directly downhill
- But sometimes this approach can lead you to areas that are too complex or "overfit" to your training data
- Regularization adds "penalty zones" to discourage overly complex solutions
- Proximal gradient descent helps navigate these penalty zones effectively

What this optimizer does:

1. Takes a step in the direction that reduces prediction error (like regular gradient descent)
2. Then takes a "proximal step" that handles the regularization penalties separately
3. By splitting the process this way, it can find solutions that balance accuracy and simplicity

Think of it like training a dog:

- The gradient step teaches the dog to complete a task correctly
- The proximal step ensures the dog doesn't develop bad habits along the way
- Together, they produce well-behaved, effective results

This approach is particularly useful when you want your model to:

- Use only a subset of available features (sparsity)
- Group related features together
- Avoid extreme parameter values

This class lets you configure how this specialized optimization process works.

## How It Works

Proximal Gradient Descent is an extension of standard gradient descent that is particularly effective
for solving optimization problems with regularization terms. It alternates between standard gradient steps
on the smooth part of the objective function and proximal operations on the non-smooth regularization terms.
This approach is especially valuable for problems involving L1 regularization (which promotes sparsity) or
other complex regularization schemes that are difficult to optimize with standard gradient methods. The
proximal approach helps maintain desirable properties of the regularization while ensuring stable convergence.
It is widely used in machine learning for training models where specific structural properties (like sparsity,
group structure, or low rank) are desired in the solution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProximalGradientDescentOptimizerOptions` | Initializes a new instance of the ProximalGradientDescentOptimizerOptions class with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `InnerIterations` | Gets or sets the number of inner iterations for each main optimization iteration. |
| `LearningRateDecreaseFactor` | Gets or sets the multiplicative factor for decreasing the learning rate when progress stalls or reverses. |
| `LearningRateIncreaseFactor` | Gets or sets the maximum number of iterations for the optimization process. |
| `ProximalStepSize` | Gets or sets the step size for the proximal operator component of the algorithm. |
| `RegularizationStrength` | Gets or sets the strength of the regularization term in the objective function. |

