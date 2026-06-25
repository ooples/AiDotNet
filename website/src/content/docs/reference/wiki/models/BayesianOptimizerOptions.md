---
title: "BayesianOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Bayesian optimization algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Bayesian optimization algorithm.

## For Beginners

Bayesian optimization is like a smart search strategy. Imagine you're trying to find 
the highest point in a hilly landscape while blindfolded - you can only ask for the height at specific locations. 
Instead of checking every possible spot (which would take too long), Bayesian optimization uses what it learns from 
previous measurements to make educated guesses about where the highest point might be. It balances between exploring 
new areas and focusing on promising regions, making it efficient for finding optimal solutions when each evaluation 
is time-consuming or expensive.

## How It Works

Bayesian optimization is a sequential design strategy for global optimization of black-box functions
that doesn't require derivatives. It's particularly useful for optimizing expensive-to-evaluate functions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcquisitionFunction` | Gets or sets the type of acquisition function to use. |
| `AcquisitionOptimizationSamples` | Gets or sets the number of samples used when optimizing the acquisition function. |
| `ExplorationFactor` | Gets or sets the exploration factor (kappa) used in the acquisition function. |
| `InitialSamples` | Gets or sets the number of initial random samples to evaluate before starting the optimization process. |
| `IsMaximization` | Gets or sets whether the objective should be maximized (true) or minimized (false). |
| `KernelFunction` | Gets or sets the kernel function used by the Gaussian Process model. |
| `LowerBound` | Gets or sets the lower bound for the search space. |
| `UpperBound` | Gets or sets the upper bound for the search space. |

