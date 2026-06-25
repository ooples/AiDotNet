---
title: "NeuralStressTest<T>"
description: "Neural network model for generating and evaluating stress test scenarios."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

Neural network model for generating and evaluating stress test scenarios.

## For Beginners

Neural Stress Testing learns to predict how a portfolio behaves under
extreme market conditions. Unlike traditional stress testing which uses fixed historical scenarios,
this model can generate new, plausible crisis scenarios (like a "Deep Fake" market crash)
to test portfolio resilience against events that haven't happened yet.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralStressTest` | Initializes a new instance of the NeuralStressTest model. |
| `NeuralStressTest(NeuralNetworkArchitecture<>,String,NeuralStressTestOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance from ONNX. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` | Adjusts action based on stress test results. |
| `CalculateRisk(Tensor<>)` | Calculates aggregate risk across generated stress scenarios. |
| `CreateNewInstance` | Creates a new instance of the NeuralStressTest model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for NeuralStressTest. |
| `StressTest(Tensor<>,Tensor<>)` | Generates stress scenarios for the given input state. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

