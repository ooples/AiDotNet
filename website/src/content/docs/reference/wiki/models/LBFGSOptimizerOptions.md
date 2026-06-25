---
title: "LBFGSOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer, which is an efficient optimization algorithm for training machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer,
which is an efficient optimization algorithm for training machine learning models.

## For Beginners

L-BFGS is an advanced optimization algorithm that helps train machine learning
models more efficiently than simpler methods like gradient descent.

Think of training a machine learning model as finding the lowest point in a hilly landscape, where the
lowest point represents the best model parameters. While basic algorithms like gradient descent simply
follow the steepest downhill path, L-BFGS is smarter:

- It remembers information about previous steps to make better decisions about where to go next
- It can take larger steps when appropriate, potentially finding the lowest point faster
- It requires less memory than some other advanced methods, making it practical for larger models

L-BFGS is particularly useful when:

- You have many parameters to optimize (complex models)
- You need faster convergence than gradient descent provides
- You have limited memory resources compared to what full second-order methods would require

This class lets you configure how L-BFGS behaves during training, including how much history it
remembers and how it adjusts its learning rate.

## How It Works

L-BFGS is a quasi-Newton optimization method that approximates the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
algorithm using limited memory. It's particularly effective for optimizing parameters in models with
many parameters, as it doesn't need to store the full Hessian matrix. This makes it more memory-efficient
than full BFGS while still providing good convergence properties.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LBFGSOptimizerOptions` | Initializes a new instance of the LBFGSOptimizerOptions class with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for gradient computation. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the L-BFGS algorithm, which controls the initial step size during optimization. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate is decreased when the algorithm encounters difficulties or needs to take more careful steps. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate is increased when the algorithm determines that larger steps would be beneficial. |
| `MaxLearningRate` | Gets or sets the maximum learning rate allowed during optimization, preventing the learning rate from becoming too large. |
| `MemorySize` | Gets or sets the memory size, which determines how many previous iterations' information the L-BFGS algorithm stores to approximate the Hessian matrix. |
| `MinLearningRate` | Gets or sets the minimum learning rate allowed during optimization, preventing the learning rate from becoming too small. |

