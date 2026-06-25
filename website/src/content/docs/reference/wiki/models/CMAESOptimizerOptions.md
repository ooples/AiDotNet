---
title: "CMAESOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm.

## For Beginners

CMA-ES is like a smart search algorithm that tries to find the best solution 
to a complex problem. Imagine you're looking for the lowest point in a mountain range with fog everywhere - 
you can only see what's right around you. CMA-ES works by sending out multiple "scouts" (solutions) in different 
directions, then learning from their findings to decide where to search next. It's particularly good at handling 
tricky landscapes where simple "always go downhill" approaches would get stuck. This class lets you configure 
how that search process works.

## How It Works

CMA-ES is a powerful evolutionary algorithm for difficult non-linear, non-convex optimization problems.
It adapts its search strategy during optimization by learning dependencies between variables and
step sizes, making it effective for complex problems where other methods might fail.

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialStepSize` | Gets or sets the initial step size that controls how far the algorithm explores from the starting point. |
| `MaxGenerations` | Gets or sets the maximum number of generations (iterations) the algorithm will run. |
| `PopulationSize` | Gets or sets the number of candidate solutions evaluated in each generation. |
| `StopTolerance` | Gets or sets the convergence threshold that determines when the algorithm should stop. |

