---
title: "DeepEnsemble<T>"
description: "Implements Deep Ensembles for uncertainty estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks`

Implements Deep Ensembles for uncertainty estimation.

## For Beginners

Deep Ensembles is one of the most effective methods for uncertainty estimation.

The concept is simple: Train multiple independent neural networks (with different random initializations)
on the same task, then use them all to make predictions. The diversity in their predictions gives you
uncertainty estimates.

Think of it like a panel of doctors giving diagnoses:

- If all doctors agree on the diagnosis, confidence is high
- If doctors give different diagnoses, uncertainty is high

Advantages:

- Very reliable uncertainty estimates
- No special training procedures needed
- Each network can use standard architectures

Disadvantages:

- Requires training and storing multiple networks
- Slower inference (must run all networks)
- Higher memory usage

Research shows that ensembles of just 5 networks often outperform more complex Bayesian methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepEnsemble(List<INeuralNetwork<>>)` | Initializes a new instance of the DeepEnsemble class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnsembleSize` | Gets the number of models in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMean(List<Tensor<>>)` | Computes the mean of multiple predictions. |
| `ComputeVariance(List<Tensor<>>,Tensor<>)` | Computes the variance of multiple predictions. |
| `EstimateAleatoricUncertainty(Tensor<>)` | Estimates aleatoric uncertainty from the ensemble. |
| `EstimateEpistemicUncertainty(Tensor<>)` | Estimates epistemic uncertainty from the ensemble. |
| `GetAllPredictions(Tensor<>)` | Gets the predictions from all ensemble members. |
| `PredictWithUncertainty(Tensor<>)` | Predicts output with uncertainty estimates from the ensemble. |

