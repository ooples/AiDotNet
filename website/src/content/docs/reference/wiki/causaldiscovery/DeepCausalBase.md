---
title: "DeepCausalBase<T>"
description: "Base class for deep learning-based causal discovery algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.DeepLearning`

Base class for deep learning-based causal discovery algorithms.

## For Beginners

These methods use neural networks to discover causal relationships.
They can capture complex nonlinear effects but require more data and computation than
traditional methods. Think of them as "letting the neural network figure out which
variables cause which" by training it on the data.

## How It Works

Deep learning methods learn causal structure by training neural networks that
parameterize the structural equation model. The DAG constraint is typically
enforced through continuous relaxation (e.g., NOTEARS-style) during training.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `DefaultKlWeight` | Default KL divergence weight for variational regularization. |
| `EdgeThreshold` | Edge weight threshold for post-training pruning. |
| `HiddenUnits` | Number of hidden units in neural network layers. |
| `InitialLogVariance` | Initial log-variance for variational parameters. |
| `LearningRate` | Learning rate for gradient-based optimization. |
| `MaxEpochs` | Maximum training epochs. |
| `MaxKlWeight` | Maximum KL divergence weight after warm-up schedule. |
| `MaxPenaltyValue` | Maximum penalty parameter (rho_max) for augmented Lagrangian methods. |
| `UseKlWarmUp` | Whether to use KL weight warm-up schedule to prevent posterior collapse. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDeepOptions(CausalDiscoveryOptions)` | Applies deep learning options. |
| `BuildFinalAdjacency(Double[0:,0:],Matrix<>,Int32)` | Builds the final weighted adjacency matrix from learned edge probabilities and covariance. |

