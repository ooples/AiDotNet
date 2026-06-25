---
title: "NormalOptimizer"
description: "Implements a normal optimization algorithm with adaptive parameters."
section: "Reference"
---

_Optimizers_

Implements a normal optimization algorithm with adaptive parameters.

## For Beginners

Imagine you're trying to find the highest peak in a mountain range, but you can't see very far.
This optimizer is like a hiker who starts at random spots, climbs to the nearest peak, and then jumps to another random spot.
The hiker learns from each climb and adjusts their strategy (like how far to jump or how carefully to look around) based on whether they're finding higher peaks or not.

## How It Works

The NormalOptimizer uses a combination of random search and adaptive parameter tuning to find optimal solutions.
It incorporates elements from genetic algorithms but operates on a single solution at a time.

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
    .ConfigureOptimizer(new NormalOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NormalOptimizer.");
```

