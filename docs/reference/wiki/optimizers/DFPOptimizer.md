---
title: "DFPOptimizer"
description: "Implements the Davidon-Fletcher-Powell (DFP) optimization algorithm for numerical optimization problems."
section: "Reference"
---

_Optimizers_

Implements the Davidon-Fletcher-Powell (DFP) optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer is like a smart navigator that learns from its past steps to make better decisions about which direction to move in the future. It's particularly good at handling complex optimization problems where the landscape of possible solutions is intricate.

## How It Works

The DFP algorithm is a quasi-Newton method for solving unconstrained nonlinear optimization problems. It approximates the inverse Hessian matrix to determine the search direction, combining the efficiency of Newton's method with the stability of gradient descent.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(0);
var trainX = new Tensor<double>(new[] { 32, 8 });
var trainY = new Tensor<double>(new[] { 32, 2 });
for (int i = 0; i < 32; i++)
{
    for (int j = 0; j < 8; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 2 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 8, numClasses: 2, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new DFPOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with DFPOptimizer.");
```

