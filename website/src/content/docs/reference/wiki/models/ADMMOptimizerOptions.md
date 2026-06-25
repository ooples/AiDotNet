---
title: "ADMMOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Alternating Direction Method of Multipliers (ADMM) optimization algorithm, which is particularly effective for problems with complex regularization requirements."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Alternating Direction Method of Multipliers (ADMM) optimization algorithm,
which is particularly effective for problems with complex regularization requirements.

## For Beginners

ADMM is like solving a complex puzzle by breaking it into smaller pieces.
Instead of trying to solve everything at once, it tackles different parts of the problem separately and then
combines the solutions. This approach is particularly good when you want your AI model to be both accurate and simple
(avoiding unnecessary complexity). Think of it as a team of specialists working together rather than one person
trying to do everything.

## How It Works

ADMM is an advanced optimization algorithm that breaks complex problems into smaller, more manageable subproblems.
It's especially useful for large-scale distributed optimization and problems with L1/L2 regularization.

## Properties

| Property | Summary |
|:-----|:--------|
| `AbsoluteTolerance` | Gets or sets the convergence tolerance that determines when the algorithm should stop. |
| `AdaptiveRhoDecrease` | Gets or sets the factor by which to decrease Rho when dual residuals are much larger than primal residuals. |
| `AdaptiveRhoFactor` | Gets or sets the factor used to determine when to adjust the adaptive Rho value. |
| `AdaptiveRhoIncrease` | Gets or sets the factor by which to increase Rho when primal residuals are much larger than dual residuals. |
| `BatchSize` | Gets or sets the batch size for mini-batch processing within each ADMM iteration. |
| `DecompositionType` | Gets or sets the type of matrix decomposition to use when solving linear systems within the ADMM algorithm. |
| `ElasticNetMixing` | Gets or sets the mixing parameter for ElasticNet regularization, balancing L1 and L2 penalties. |
| `RegularizationStrength` | Gets or sets the strength of the regularization penalty. |
| `RegularizationType` | Gets or sets the type of regularization to apply to the optimization problem. |
| `Rho` | Gets or sets the penalty parameter that controls the balance between the original objective and the constraint satisfaction. |
| `UseAdaptiveRho` | Gets or sets whether the algorithm should automatically adjust the Rho parameter during optimization. |

