---
title: "BFGSOptimizer"
description: "Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm.

## For Beginners

BFGS is an advanced optimization algorithm that tries to find the best solution
by making smart steps based on the function's behavior. It's particularly good at handling complex problems
where the function being optimized is smooth but potentially has many variables.

## How It Works

BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
It approximates the Hessian matrix of second derivatives of the function to be minimized.

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
    .ConfigureOptimizer(new BFGSOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with BFGSOptimizer.");
```

