---
title: "InverseProblemResult<T>"
description: "Results from inverse problem optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.Interfaces`

Results from inverse problem optimization.

## Properties

| Property | Summary |
|:-----|:--------|
| `Converged` | Whether the optimization converged. |
| `DataLoss` | Final data loss (fit to observations). |
| `IterationsToConverge` | Number of iterations until convergence. |
| `ParameterCorrelations` | Correlation matrix between parameters (for uncertainty analysis). |
| `ParameterHistory` | History of parameter values during training. |
| `ParameterNames` | Names of the identified parameters. |
| `ParameterUncertainties` | Estimated uncertainties (standard deviations) for each parameter. |
| `Parameters` | The identified parameter values. |
| `PhysicsLoss` | Final physics loss (PDE residual). |
| `TotalLoss` | Total loss (data + physics + regularization). |

