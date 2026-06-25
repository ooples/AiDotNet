---
title: "DifferentialEvolutionOptimizer"
description: "Implements the Differential Evolution optimization algorithm for numerical optimization problems."
section: "Reference"
---

_Optimizers_

Implements the Differential Evolution optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer works by evolving a population of candidate solutions over time. It's inspired by biological evolution and is good at finding global optima in complex problem spaces.

## How It Works

Differential Evolution is a population-based optimization algorithm that is particularly well-suited for solving non-linear, non-differentiable continuous space functions. It's known for its simplicity, robustness, and effectiveness in various optimization scenarios.

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
    .ConfigureOptimizer(new DifferentialEvolutionOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with DifferentialEvolutionOptimizer.");
```

