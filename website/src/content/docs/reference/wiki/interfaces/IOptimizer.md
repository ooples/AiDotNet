---
title: "IOptimizer<T, TInput, TOutput>"
description: "Defines the contract for optimization algorithms used in machine learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for optimization algorithms used in machine learning models.

## How It Works

An optimizer is responsible for finding the best parameters for a machine learning model
by minimizing or maximizing an objective function.

**For Beginners:** Think of an optimizer as a "tuning expert" that adjusts your model's settings
to make it perform better. Just like tuning a radio to get the clearest signal, an optimizer
tunes your model's parameters to get the best predictions.

Common examples of optimizers include:

- Gradient Descent: Gradually moves toward better parameters by following the slope
- Adam: An advanced optimizer that adapts its learning rate for each parameter
- L-BFGS: Works well for smaller datasets and uses memory of previous steps

Why optimizers matter:

- They determine how quickly your model learns
- They affect whether your model finds the best solution or gets stuck
- Different optimizers work better for different types of problems

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` | Gets the configuration options for the optimization algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process to find the best parameters for a model. |
| `Reset` | Resets the optimizer state to prepare for a fresh optimization run. |
| `SetModel(IFullModel<,,>)` | Sets the model that this optimizer will optimize. |
| `ShouldEarlyStop` | Determines whether the optimization process should stop early. |

