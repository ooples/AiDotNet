---
title: "InverseProblemOptions<T>"
description: "Configuration options for inverse problem PINN training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.PhysicsInformed.Interfaces`

Configuration options for inverse problem PINN training.

## Properties

| Property | Summary |
|:-----|:--------|
| `DataWeight` | Weight for the observation data loss relative to physics loss. |
| `EstimateUncertainty` | Whether to estimate parameter uncertainty. |
| `LogParameterHistory` | Whether to log parameter estimates during training. |
| `ParameterLearningRate` | Learning rate for the unknown parameters (if separate rates are used). |
| `PriorMeans` | Prior means for Bayesian regularization. |
| `PriorStandardDeviations` | Prior standard deviations for Bayesian regularization. |
| `Regularization` | The type of regularization to apply. |
| `RegularizationStrength` | Regularization strength (λ in the formulas above). |
| `UncertaintySamples` | Number of samples for uncertainty estimation (if enabled). |
| `UseSeparateLearningRates` | Whether to use separate learning rates for solution and parameters. |

