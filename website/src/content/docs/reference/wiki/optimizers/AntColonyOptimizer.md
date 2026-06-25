---
title: "AntColonyOptimizer"
description: "Implements the Ant Colony Optimization algorithm for solving optimization problems."
section: "Reference"
---

_Optimizers_

Implements the Ant Colony Optimization algorithm for solving optimization problems.

## For Beginners

Think of this algorithm as a group of ants searching for the best path to food.
Each ant leaves a trail (pheromone) that other ants can follow. Over time, the best paths get stronger trails,
leading to better solutions.

## How It Works

Ant Colony Optimization is inspired by the behavior of ants in finding paths between their colony and food sources.
It uses virtual "ants" to explore the solution space and find optimal solutions.

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
    .ConfigureOptimizer(new AntColonyOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AntColonyOptimizer.");
```

