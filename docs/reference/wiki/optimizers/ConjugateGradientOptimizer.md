---
title: "ConjugateGradientOptimizer"
description: "Implements the Conjugate Gradient optimization algorithm for numerical optimization problems."
section: "Reference"
---

_Optimizers_

Implements the Conjugate Gradient optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer is like a smart hiker trying to find the lowest point in a hilly landscape. It uses information about the slope (gradient) and its previous steps to decide on the best direction to move next, allowing it to find the lowest point (optimal solution) more efficiently than simpler methods.

## How It Works

The Conjugate Gradient method is an algorithm for the numerical solution of particular systems of linear equations, namely those whose matrix is symmetric and positive-definite. It is often used to solve unconstrained optimization problems such as energy minimization.

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
    .ConfigureOptimizer(new ConjugateGradientOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ConjugateGradientOptimizer.");
```

