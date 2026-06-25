---
title: "BFGSOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm.

## For Beginners

BFGS is an advanced optimization algorithm that helps find the best solution 
to a problem efficiently. Think of it like finding the lowest point in a hilly landscape when you can only 
see the steepness at your current position. Unlike simpler methods that just go downhill, BFGS builds a 
"mental map" of the landscape as it explores, helping it make smarter decisions about where to go next. 
This makes it faster and more reliable for complex problems.

## How It Works

The BFGS algorithm is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
It approximates the Hessian matrix (which contains second derivatives) using gradient information,
making it more efficient than methods that require explicit computation of the Hessian.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BFGSOptimizerOptions` | Initializes a new instance of the BFGSOptimizerOptions class with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for gradient computation. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the optimization process. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which to decrease the learning rate when progress is poor. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which to increase the learning rate when progress is good. |
| `MaxLearningRate` | Gets or sets the maximum allowed learning rate during optimization. |
| `MinLearningRate` | Gets or sets the minimum allowed learning rate during optimization. |

