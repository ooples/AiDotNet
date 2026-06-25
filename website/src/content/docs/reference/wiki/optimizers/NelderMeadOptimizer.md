---
title: "NelderMeadOptimizer"
description: "Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method."
section: "Reference"
---

_Optimizers_

Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method.

## For Beginners

Imagine you're trying to find the lowest point in a hilly landscape. The Nelder-Mead method is like
having a group of explorers who work together, moving and reshaping their search pattern to find the lowest point.
They don't need to know which way is downhill; they just compare their positions and adjust accordingly.

## How It Works

The Nelder-Mead method is a heuristic search method that can optimize a problem with N variables.
It attempts to minimize a scalar-valued nonlinear function of n real variables using only function values,
without any derivative information.

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
    .ConfigureOptimizer(new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NelderMeadOptimizer.");
```

