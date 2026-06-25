---
title: "AttentionAllocation<T>"
description: "Portfolio optimizer using Multi-Head Attention to capture asset relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Portfolio`

Portfolio optimizer using Multi-Head Attention to capture asset relationships.

## For Beginners

Uses the same "attention" mechanism that powers ChatGPT to look at
relationships between different stocks. It learns which assets move together or diverge
dynamically based on market context, rather than relying on static historical correlations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AttentionAllocation` | Initializes a new instance of the AttentionAllocation model. |
| `AttentionAllocation(NeuralNetworkArchitecture<>,String,AttentionAllocationOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance from ONNX. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the AttentionAllocation model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the AttentionAllocation. |
| `OptimizePortfolio(Tensor<>)` | Optimizes portfolio weights using attention mechanism. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

