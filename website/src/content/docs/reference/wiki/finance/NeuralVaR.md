---
title: "NeuralVaR<T>"
description: "Neural Value-at-Risk (VaR) model for non-linear market risk assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

Neural Value-at-Risk (VaR) model for non-linear market risk assessment.

## For Beginners

Value-at-Risk (VaR) is a way to answer the question:
"What is the most I could lose on this investment tomorrow with 95% confidence?"
Traditional methods often assume simple patterns, but this AI model "learns"
from historical market crashes and complex trends to give a more realistic
estimate of risk.

## How It Works

NeuralVaR uses deep neural networks to estimate the potential loss of a portfolio
under various market conditions, accounting for complex non-linear dependencies.

Reference: Riskfuel, "Neural Value at Risk", 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralVaR(NeuralNetworkArchitecture<>,NeuralVaROptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a NeuralVaR model in native mode for training. |
| `NeuralVaR(NeuralNetworkArchitecture<>,String,NeuralVaROptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a NeuralVaR model using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` |  |
| `CalculateRisk(Tensor<>)` |  |
| `CreateNewInstance` | Executes CreateNewInstance for the NeuralVaR. |
| `GetModelMetadata` | Executes GetModelMetadata for the NeuralVaR. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the NeuralVaR. |
| `PredictCore(Tensor<>)` | Executes Predict for the NeuralVaR. |
| `Train(Tensor<>,Tensor<>)` | Executes Train for the NeuralVaR. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the NeuralVaR. |

