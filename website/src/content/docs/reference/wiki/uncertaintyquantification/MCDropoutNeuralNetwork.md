---
title: "MCDropoutNeuralNetwork<T>"
description: "Implements Monte Carlo Dropout for uncertainty estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks`

Implements Monte Carlo Dropout for uncertainty estimation.

## For Beginners

MC Dropout is the simplest way to add uncertainty estimation to existing neural networks.

The idea is straightforward:

1. Add dropout layers to your network
2. Keep dropout active even during prediction (normally it's turned off)
3. Run multiple predictions with different random dropout patterns
4. The variation in predictions tells you the uncertainty

This is much easier than full Bayesian neural networks but still provides useful uncertainty estimates.
It's like getting a second (and third, and fourth...) opinion from slightly different versions of your model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MCDropoutNeuralNetwork(NeuralNetworkArchitecture<>,Int32)` | Initializes a new instance of the MCDropoutNeuralNetwork class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMean(List<Tensor<>>)` | Computes the mean of multiple predictions. |
| `ComputeVariance(List<Tensor<>>,Tensor<>)` | Computes the variance of multiple predictions. |
| `EnableMCMode(Boolean)` | Enables or disables Monte Carlo mode for all MC dropout layers. |
| `EstimateAleatoricUncertainty(Tensor<>)` | Estimates aleatoric uncertainty. |
| `EstimateEpistemicUncertainty(Tensor<>)` | Estimates epistemic uncertainty. |
| `PredictWithUncertainty(Tensor<>)` | Predicts output with uncertainty estimates using MC dropout. |

