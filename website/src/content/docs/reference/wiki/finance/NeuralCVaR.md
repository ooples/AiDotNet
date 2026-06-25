---
title: "NeuralCVaR<T>"
description: "A neural network model for estimating Conditional Value at Risk (CVaR), also known as Expected Shortfall."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

A neural network model for estimating Conditional Value at Risk (CVaR), also known as Expected Shortfall.

## For Beginners

CVaR (Conditional Value at Risk) answers the question: "If things get really bad (worse than VaR),
how much will I lose on average?"

While VaR (Value at Risk) gives you a threshold (e.g., "I'm 95% sure I won't lose more than $100"),
CVaR tells you the average loss in that remaining 5% of worst-case scenarios (e.g., "$150").

This model uses a neural network to predict this value based on market conditions. It's often considered
a better risk measure than VaR because it accounts for the magnitude of extreme losses ("fat tails").

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralCVaR` | Initializes a new instance with default architecture settings. |
| `NeuralCVaR(NeuralNetworkArchitecture<>,NeuralCVaROptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the NeuralCVaR model. |
| `NeuralCVaR(NeuralNetworkArchitecture<>,String,NeuralCVaROptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the NeuralCVaR model from a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` | Adjusts a proposed action to satisfy CVaR constraints. |
| `CalculateRisk(Tensor<>)` | Calculates the Conditional Value at Risk for the given input. |
| `CalculateVaR(Tensor<>,Tensor<>)` | Calculates VaR as a byproduct of CVaR estimation (using Gaussian assumption or auxiliary output). |
| `CreateNewInstance` | Creates a new instance of the NeuralCVaR model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for NeuralCVaR. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

