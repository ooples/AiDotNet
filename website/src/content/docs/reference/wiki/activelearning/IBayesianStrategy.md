---
title: "IBayesianStrategy<T, TInput, TOutput>"
description: "Interface for Bayesian query strategies (e.g., BALD)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for Bayesian query strategies (e.g., BALD).

## Properties

| Property | Summary |
|:-----|:--------|
| `MonteCarloSamples` | Gets the number of Monte Carlo samples for uncertainty estimation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMutualInformation(IFullModel<,,>,)` | Computes the mutual information between predictions and model parameters. |
| `ComputePredictiveEntropy(IFullModel<,,>,)` | Computes the predictive entropy for a sample. |

