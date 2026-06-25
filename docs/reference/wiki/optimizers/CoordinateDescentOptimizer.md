---
title: "CoordinateDescentOptimizer"
description: "Implements the Coordinate Descent optimization algorithm for numerical optimization problems."
section: "Reference"
---

_Optimizers_

Implements the Coordinate Descent optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer is like adjusting the knobs on a complex machine one at a time. 
It focuses on improving one aspect of the solution at a time, which can be more manageable and sometimes 
more effective than trying to adjust everything at once.

## How It Works

Coordinate Descent is an optimization algorithm that minimizes a multivariable function by solving a series of 
single-variable optimization problems. It cycles through each variable (coordinate) and optimizes it while holding 
the others constant.

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
    .ConfigureOptimizer(new CoordinateDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with CoordinateDescentOptimizer.");
```

