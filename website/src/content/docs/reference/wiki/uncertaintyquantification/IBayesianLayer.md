---
title: "IBayesianLayer<T>"
description: "Defines the contract for Bayesian neural network layers that support probabilistic inference."
section: "API Reference"
---

`Interfaces` · `AiDotNet.UncertaintyQuantification.Interfaces`

Defines the contract for Bayesian neural network layers that support probabilistic inference.

## For Beginners

A Bayesian layer is different from a regular neural network layer because
instead of having fixed weights, it has distributions over weights.

Think of regular weights as saying "the connection strength is exactly 2.5", while Bayesian weights
say "the connection strength is probably around 2.5, but could be anywhere from 2.0 to 3.0".

This probabilistic approach allows the network to express uncertainty in its predictions,
which is crucial for safety-critical applications.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddKLDivergenceGradients()` | Adds the KL divergence gradients (regularization term) into the layer's accumulated gradients. |
| `GetKLDivergence` | Gets the KL divergence term for variational inference. |
| `SampleWeights` | Samples from the weight distribution for stochastic forward passes. |

