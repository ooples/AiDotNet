---
title: "GradientHelperFactory<T>"
description: "Factory methods for creating gradient helpers from various model types."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability.Helpers`

Factory methods for creating gradient helpers from various model types.

## For Beginners

This factory makes it easy to get gradient computation for
any type of model. Just pass your model and it figures out the best way to
compute gradients.

The factory supports:

- Neural networks (uses efficient backpropagation)
- Autodiff models (uses GradientTape)
- Any prediction function (uses numerical gradients)

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateGradientFunction(IFullModel<,Tensor<>,Tensor<>>)` | Creates a gradient function for any IFullModel. |
| `FromAutodiffModel(Func<ComputationNode<>,ComputationNode<>>)` | Creates a gradient helper from an autodiff model function. |
| `FromNeuralNetwork(INeuralNetwork<>)` | Creates a gradient helper for a neural network. |
| `FromPredictFunction(Func<Vector<>,Vector<>>,Double)` | Creates a gradient helper from a prediction function. |

