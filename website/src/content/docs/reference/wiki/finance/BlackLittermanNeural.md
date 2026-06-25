---
title: "BlackLittermanNeural<T>"
description: "Neural network adaptation of the Black-Litterman model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Portfolio`

Neural network adaptation of the Black-Litterman model.

## For Beginners

The Black-Litterman model combines "market equilibrium" (what everyone thinks)
with "investor views" (your specific predictions). This neural version learns to generate
these "views" and "confidence levels" from data automatically, blending them with the
market baseline to produce robust portfolios.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlackLittermanNeural` | Initializes a new instance of the BlackLittermanNeural model. |
| `BlackLittermanNeural(NeuralNetworkArchitecture<>,String,BlackLittermanNeuralOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance from ONNX. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the BlackLittermanNeural model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the BlackLittermanNeural. |
| `OptimizePortfolio(Tensor<>)` | Optimizes portfolio weights by generating views and blending with equilibrium. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

