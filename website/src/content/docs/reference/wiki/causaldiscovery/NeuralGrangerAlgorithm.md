---
title: "NeuralGrangerAlgorithm<T>"
description: "Neural Granger Causality — deep learning extension of Granger causality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

Neural Granger Causality — deep learning extension of Granger causality.

## For Beginners

Standard Granger causality assumes linear relationships.
Neural Granger uses neural networks instead, so it can find nonlinear causal
relationships. For example, "X causes Y, but only when X is in a certain range."

## How It Works

Neural Granger Causality replaces the linear VAR model in Granger causality with
a multi-layer perceptron, combined with group-lasso sparsity penalty on the input
layer weights. The L2 norm of the first-layer weights for each input variable's
lags indicates causal strength: ||W1[:,i*MaxLag:(i+1)*MaxLag]||_2.

**Algorithm:**

- For each target j, build lagged feature matrix X with all variables' lags
- Train MLP: X → x_j[t] with sigmoid activations
- Apply group-lasso penalty: lambda * sum_i ||W1[:,i_lags]||_2
- After training, causal strength from i to j = ||W1[:,i_lags]||_2
- Threshold to get final graph

Reference: Tank et al. (2021), "Neural Granger Causality", IEEE TPAMI.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

