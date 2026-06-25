---
title: "BayesianDenseLayer<T>"
description: "Implements a Bayesian dense (fully-connected) layer using variational inference."
section: "API Reference"
---

`Layers` · `AiDotNet.UncertaintyQuantification.Layers`

Implements a Bayesian dense (fully-connected) layer using variational inference.

## For Beginners

A Bayesian Dense Layer is similar to a regular dense layer, but instead
of having fixed weights, it learns probability distributions over weights.

This is based on the "Bayes by Backprop" algorithm which uses variational inference to
approximate the true posterior distribution of weights.

The layer maintains two sets of parameters for each weight:

- Mean (μ): The average value of the weight
- Standard deviation (σ): How much the weight varies

During forward passes, weights are sampled from these distributions, allowing the network
to express uncertainty in its predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianDenseLayer(Int32,Int32,Double,Nullable<Int32>)` | Initializes a new instance of the BayesianDenseLayer class. |
| `BayesianDenseLayer(Int32,Int32,IActivationFunction<>,Double,Nullable<Int32>)` | Initializes a new instance of the `BayesianDenseLayer` class with a custom activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddKLDivergenceGradients()` |  |
| `ClearGradients` |  |
| `Forward(Tensor<>)` | Performs the forward pass using sampled weights. |
| `GetKLDivergence` | Computes the KL divergence between the weight distribution and the prior. |
| `GetParameters` | Gets all trainable parameters. |
| `ResetState` | Resets the internal state of the layer. |
| `SampleWeights` | Samples weights from the learned distributions. |
| `SetParameters(Vector<>)` | Sets all trainable parameters. |
| `UpdateParameters()` | Updates parameters using the accumulated gradients. |
| `UpdateParameters(Vector<>)` |  |

