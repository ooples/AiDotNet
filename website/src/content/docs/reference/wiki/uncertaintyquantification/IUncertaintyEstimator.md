---
title: "IUncertaintyEstimator<T>"
description: "Defines the contract for models that can estimate prediction uncertainty."
section: "API Reference"
---

`Interfaces` · `AiDotNet.UncertaintyQuantification.Interfaces`

Defines the contract for models that can estimate prediction uncertainty.

## For Beginners

Uncertainty estimation helps you understand how confident a model is in its predictions.

Think of it like a weather forecast that not only predicts rain but also tells you how sure it is:

- "90% chance of rain" shows high confidence
- "50% chance of rain" shows high uncertainty

This interface is for models that can provide both a prediction and an estimate of how uncertain that prediction is.
This is crucial for safety-critical applications like medical diagnosis or autonomous vehicles.

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateAleatoricUncertainty(Tensor<>)` | Estimates aleatoric uncertainty (data noise) for the given input. |
| `EstimateEpistemicUncertainty(Tensor<>)` | Estimates epistemic uncertainty (model uncertainty) for the given input. |
| `PredictWithUncertainty(Tensor<>)` | Predicts output with uncertainty estimates for a single input. |

