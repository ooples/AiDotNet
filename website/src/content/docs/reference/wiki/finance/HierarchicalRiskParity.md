---
title: "HierarchicalRiskParity<T>"
description: "Neural Hierarchical Risk Parity (HRP) for portfolio optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Portfolio`

Neural Hierarchical Risk Parity (HRP) for portfolio optimization.

## For Beginners

HRP is a modern portfolio construction technique that uses machine learning
to cluster similar assets together and then allocates risk equally across these clusters.
Unlike traditional Mean-Variance Optimization, it doesn't require inverting a covariance matrix,
making it robust to noise and stable even when assets are highly correlated.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HierarchicalRiskParity` | Initializes a new instance of the HierarchicalRiskParity model. |
| `HierarchicalRiskParity(NeuralNetworkArchitecture<>,String,HierarchicalRiskParityOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance from ONNX. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the HierarchicalRiskParity model with the same configuration. |
| `Forward(Tensor<>)` | Executes a forward pass through the network layers. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the HierarchicalRiskParity. |
| `OptimizePortfolio(Tensor<>)` | Optimizes portfolio weights using HRP logic. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

