---
title: "TrustRegionOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for Trust Region optimization algorithms, which are robust methods for solving nonlinear optimization problems."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Trust Region optimization algorithms, which are robust methods for
solving nonlinear optimization problems.

## For Beginners

Trust Region methods are like exploring with a map that's only accurate near your current location.

When solving optimization problems:

- We often use approximations of the objective function to determine the next step
- But these approximations are only reliable within a certain distance

Trust Region methods solve this by:

- Creating a simplified model of the function around the current point
- Defining a "trust region" where this model is considered reliable
- Finding the best step within this region
- Adjusting the region size based on how well the model predicts actual function behavior

This approach offers several benefits:

- More robust than many other methods, especially for difficult problems
- Handles non-convex functions well
- Can make progress even when the Hessian is not positive definite
- Naturally limits step sizes to prevent erratic behavior

This class lets you configure how the Trust Region algorithm behaves.

## How It Works

Trust Region methods are iterative optimization techniques designed to find local minima or maxima of objective 
functions, particularly in nonlinear problems. Unlike line search methods, which determine the step size along 
a predefined direction, trust region methods concurrently optimize both the direction and the magnitude of the 
step within a specified neighborhood (the "trust region") around the current iterate. The central idea is to 
construct a simplified model—often a quadratic approximation—that represents the objective function near the 
current point. This model serves as a surrogate, guiding the search for the optimum within a bounded region. 
The size of the trust region is dynamically adjusted based on how well the model predicts the actual behavior 
of the objective function. This class inherits from GradientBasedOptimizerOptions and adds parameters specific 
to Trust Region optimization.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcceptanceThreshold` | Gets or sets the threshold for accepting a step. |
| `AdaptationRate` | Gets or sets the rate at which the trust region radius adapts to new information. |
| `BatchSize` | Gets or sets the batch size for gradient computation. |
| `CGTolerance` | Gets or sets the convergence tolerance for the Conjugate Gradient method used to solve the trust region subproblem. |
| `ContractionFactor` | Gets or sets the factor by which to contract the trust region radius after an unsuccessful step. |
| `ExpansionFactor` | Gets or sets the factor by which to expand the trust region radius after a very successful step. |
| `InitialTrustRegionRadius` | Gets or sets the initial radius of the trust region. |
| `MaxCGIterations` | Gets or sets the maximum number of iterations for the Conjugate Gradient method used to solve the trust region subproblem. |
| `MaxTrustRegionRadius` | Gets or sets the maximum allowed radius of the trust region. |
| `MinTrustRegionRadius` | Gets or sets the minimum allowed radius of the trust region. |
| `UnsuccessfulThreshold` | Gets or sets the threshold for considering a step unsuccessful. |
| `UseAdaptiveTrustRegionRadius` | Gets or sets a value indicating whether to use adaptive trust region radius adjustment. |
| `VerySuccessfulThreshold` | Gets or sets the threshold for considering a step very successful. |

