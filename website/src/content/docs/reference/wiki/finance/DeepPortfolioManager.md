---
title: "DeepPortfolioManager<T>"
description: "Deep Portfolio Manager for end-to-end portfolio weight optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Portfolio`

Deep Portfolio Manager for end-to-end portfolio weight optimization.

## For Beginners

Managing a portfolio means deciding exactly what percentage 
of your money should go into each stock (e.g., 10% Apple, 5% Microsoft). 
While traditional methods use complex math formulas, this AI model looks 
at historical performance and current trends to "guess" the best weights 
that will give you the most profit with the least risk.

## How It Works

DeepPortfolioManager uses deep learning to directly map market input data
to optimal portfolio weights, typically maximizing a risk-adjusted return metric.

Reference: Zhang et al., "Deep Learning for Portfolio Management", 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepPortfolioManager(NeuralNetworkArchitecture<>,DeepPortfolioManagerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepPortfolioManager in native mode for training. |
| `DeepPortfolioManager(NeuralNetworkArchitecture<>,String,DeepPortfolioManagerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DeepPortfolioManager using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the DeepPortfolioManager. |
| `GetModelMetadata` | Executes GetModelMetadata for the DeepPortfolioManager. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the DeepPortfolioManager. |
| `OptimizePortfolio(Tensor<>)` |  |
| `PredictCore(Tensor<>)` | Executes Predict for the DeepPortfolioManager. |
| `Train(Tensor<>,Tensor<>)` | Executes Train for the DeepPortfolioManager. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the DeepPortfolioManager. |

