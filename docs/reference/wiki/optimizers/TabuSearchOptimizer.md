---
title: "TabuSearchOptimizer"
description: "Represents a Tabu Search optimizer for machine learning models."
section: "Reference"
---

_Optimizers_

Represents a Tabu Search optimizer for machine learning models.

## For Beginners

Think of Tabu Search as a smart explorer:

- The explorer (optimizer) looks for the best solution in a complex landscape
- It remembers recently visited places (tabu list) to avoid going in circles
- It adapts its search strategy over time to balance between exploring new areas and refining good solutions

This method is particularly effective for problems with many local optima.

## How It Works

The TabuSearchOptimizer implements the Tabu Search algorithm, a metaheuristic search method used in optimization.
It explores the solution space by iteratively moving from one solution to the best solution in its neighborhood,
while keeping a list of recently visited solutions (the tabu list) to avoid cycling and encourage exploration of new areas.

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
    .ConfigureOptimizer(new TabuSearchOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with TabuSearchOptimizer.");
```

