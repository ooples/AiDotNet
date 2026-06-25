---
title: "MomentumOptimizer"
description: "Implements the Momentum optimization algorithm for gradient-based optimization."
section: "Reference"
---

_Optimizers_

Implements the Momentum optimization algorithm for gradient-based optimization.

## For Beginners

Imagine you're rolling a ball down a hill to find the lowest point. The Momentum optimizer is like giving
that ball some "memory" of its previous movements. This helps it move faster in consistent directions and
resist getting stuck in small bumps or divots along the way.

## How It Works

The Momentum optimizer is an extension of gradient descent that helps accelerate the optimization process
in relevant directions and dampens oscillations. It does this by adding a fraction of the update vector
of the past time step to the current update vector.

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
    .ConfigureOptimizer(new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with MomentumOptimizer.");
```

