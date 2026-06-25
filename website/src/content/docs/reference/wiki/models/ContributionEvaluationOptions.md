---
title: "ContributionEvaluationOptions"
description: "Configuration options for client contribution evaluation in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for client contribution evaluation in federated learning.

## For Beginners

These options configure how the system measures each client's
contribution to the global model. This is important for detecting free-riders, rewarding
valuable participants, and identifying low-quality or poisoned updates.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvergenceTolerance` | Gets or sets the convergence tolerance for Monte Carlo Shapley estimation. |
| `EvaluationFrequency` | Gets or sets how often to evaluate contributions (in FL rounds). |
| `FreeRiderThreshold` | Gets or sets the minimum contribution score below which a client is flagged as a free-rider. |
| `Method` | Gets or sets the contribution evaluation method. |
| `SamplingRounds` | Gets or sets the number of Monte Carlo sampling rounds for Data Shapley. |
| `UsePerformanceCache` | Gets or sets whether to use a performance cache to avoid redundant model evaluations during Shapley value computation. |

