---
title: "BayesianNeuralNetwork<T>"
description: "Implements a Bayesian Neural Network that provides uncertainty estimates with predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks`

Implements a Bayesian Neural Network that provides uncertainty estimates with predictions.

## For Beginners

A Bayesian Neural Network (BNN) is a neural network that can tell you
not just what it predicts, but also how uncertain it is about that prediction.

This is incredibly important for safety-critical applications like:

- Medical diagnosis: "This might be cancer, but I'm very uncertain - get a second opinion"
- Autonomous driving: "I'm not sure what that object is - proceed with caution"
- Financial predictions: "The market might go up, but there's high uncertainty"

The network achieves this by making multiple predictions with slightly different weights
(sampled from learned probability distributions) and analyzing how much these predictions vary.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianNeuralNetwork(NeuralNetworkArchitecture<>,Int32)` | Initializes a new instance of the BayesianNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeKLDivergence` | Computes the total KL divergence from all Bayesian layers. |
| `ComputeMean(List<Tensor<>>)` | Computes the mean of multiple predictions. |
| `ComputeVariance(List<Tensor<>>,Tensor<>)` | Computes the variance of multiple predictions. |
| `EstimateAleatoricUncertainty(Tensor<>)` | Estimates aleatoric (data) uncertainty. |
| `EstimateEpistemicUncertainty(Tensor<>)` | Estimates epistemic (model) uncertainty. |
| `InitializeLayers` |  |
| `PredictWithUncertainty(Tensor<>)` | Predicts output with uncertainty estimates. |
| `Train(Tensor<>,Tensor<>)` |  |

