---
title: "GeneticAlgorithmOptimizer"
description: "Represents a Genetic Algorithm optimizer for machine learning models."
section: "Reference"
---

_Optimizers_

Represents a Genetic Algorithm optimizer for machine learning models.

## For Beginners

Think of the Genetic Algorithm optimizer like breeding the best solutions:

- Start with a group of random solutions (like a group of different recipes)
- Test how good each solution is (like tasting each recipe)
- Choose the best solutions (like picking the tastiest recipes)
- Create new solutions by mixing the best ones (like combining ingredients from the best recipes)
- Sometimes make small random changes (like accidentally adding a new spice)
- Repeat this process many times to find the best solution (or the tastiest recipe!)

This approach is good at finding solutions for complex problems where traditional methods might struggle.

## How It Works

The Genetic Algorithm optimizer is an evolutionary optimization technique inspired by the process of natural selection.
It evolves a population of potential solutions over multiple generations to find an optimal or near-optimal solution.

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
    .ConfigureOptimizer(new GeneticAlgorithmOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with GeneticAlgorithmOptimizer.");
```

