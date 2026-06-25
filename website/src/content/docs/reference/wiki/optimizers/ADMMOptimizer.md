---
title: "ADMMOptimizer"
description: "Implements the Alternating Direction Method of Multipliers (ADMM) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Alternating Direction Method of Multipliers (ADMM) optimization algorithm.

## For Beginners

ADMM is like solving a complex puzzle by breaking it into smaller, manageable pieces.
It's particularly good at handling problems with constraints or when you want to distribute the computation across multiple processors.

## How It Works

ADMM is an algorithm for solving convex optimization problems, particularly useful for large-scale and distributed optimization.
It combines the benefits of dual decomposition and augmented Lagrangian methods.

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
    .ConfigureOptimizer(new ADMMOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ADMMOptimizer.");
```

