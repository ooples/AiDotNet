---
title: "LBFGSOptimizer"
description: "Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm.

## For Beginners

L-BFGS is an advanced optimization algorithm that efficiently finds the minimum of a function, especially useful for problems with many variables. It uses information from previous iterations to make intelligent decisions about where to search next, while keeping memory usage low.

## How It Works

L-BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems. It approximates the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using a limited amount of computer memory, making it suitable for optimization problems with many variables.

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
    .ConfigureOptimizer(new LBFGSOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LBFGSOptimizer.");
```

